import os
import json
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from config import Config
from dataset import PMCVQADataset, collate_fn
from model import (PMCVQAModel, get_fusion_head_params, get_lora_params,
                   count_trainable_params)
from explain import explain_samples


def compute_class_weights_from_csv(csv_path, num_samples=0):
    import csv
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    counts = torch.zeros(4)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if num_samples and i >= num_samples:
                break
            counts[label_map[row['Answer'].strip()]] += 1
    total = counts.sum()
    weights = total / (counts + 1e-8)
    weights = weights / weights.sum() * 4
    return weights


def train_epoch(model, loader, criterion, optimizer, scaler, scheduler, device, config):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Train', leave=False)
    for batch in pbar:
        images = batch['image'].to(device)
        q_ids = batch['question_input_ids'].to(device)
        q_mask = batch['question_attention_mask'].to(device)
        c_ids = batch['choices_input_ids'].to(device)
        c_mask = batch['choices_attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        if config.use_amp and device.type == 'cuda':
            with autocast('cuda'):
                scores = model(images, q_ids, q_mask, c_ids, c_mask)
                loss = criterion(scores, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            scores = model(images, q_ids, q_mask, c_ids, c_mask)
            loss = criterion(scores, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        preds = scores.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        pbar.set_postfix(loss=loss.item(), acc=correct / max(total, 1))

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in tqdm(loader, desc='Val', leave=False):
        images = batch['image'].to(device)
        q_ids = batch['question_input_ids'].to(device)
        q_mask = batch['question_attention_mask'].to(device)
        c_ids = batch['choices_input_ids'].to(device)
        c_mask = batch['choices_attention_mask'].to(device)
        labels = batch['label'].to(device)

        scores = model(images, q_ids, q_mask, c_ids, c_mask)
        loss = criterion(scores, labels)

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

        probs = torch.softmax(scores, dim=-1)
        all_preds.append(scores.argmax(dim=-1).cpu())
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    acc = (all_preds == all_labels).float().mean().item()

    per_class_acc = torch.zeros(4)
    for c in range(4):
        mask = all_labels == c
        per_class_acc[c] = (all_preds[mask] == all_labels[mask]).float().mean().item() if mask.any() else 0.0

    precision, recall, f1 = torch.zeros(4), torch.zeros(4), torch.zeros(4)
    for c in range(4):
        tp = ((all_preds == c) & (all_labels == c)).sum().item()
        fp = ((all_preds == c) & (all_labels != c)).sum().item()
        fn = ((all_preds != c) & (all_labels == c)).sum().item()
        precision[c] = tp / (tp + fp + 1e-8)
        recall[c] = tp / (tp + fn + 1e-8)
        f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c] + 1e-8)
    macro_f1 = f1.mean().item()

    top2_preds = all_probs.topk(2, dim=-1).indices
    top2_acc = (top2_preds == all_labels.unsqueeze(1)).any(dim=-1).float().mean().item()

    try:
        auc_roc = roc_auc_score(all_labels.numpy(), all_probs.numpy(), multi_class='ovr')
    except Exception:
        auc_roc = 0.0

    confusion = torch.zeros(4, 4, dtype=torch.int64)
    for t, p in zip(all_labels, all_preds):
        confusion[t, p] += 1

    return (total_loss / total, acc, per_class_acc.tolist(),
            precision.tolist(), recall.tolist(), f1.tolist(),
            macro_f1, top2_acc, auc_roc,
            all_labels.numpy(), all_preds.numpy(), confusion.numpy())


def main():
    config = Config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)

    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or None,
            config=vars(config),
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True)

    full_dataset = PMCVQADataset(
        csv_path=config.train_csv,
        image_dir=config.image_dir,
        tokenizer=tokenizer,
        max_length=config.max_text_length,
        image_size=config.image_size,
        split='train',
        train=True,
    )

    limit = config.max_train_samples
    if limit and limit < len(full_dataset):
        labels = [s['Answer'].strip() for s in full_dataset.samples]
        indices = list(range(len(full_dataset)))
        selected, _, _, _ = train_test_split(
            indices, labels, train_size=limit,
            stratify=labels, random_state=42,
        )
        full_dataset.samples = [full_dataset.samples[i] for i in selected]
    print(f"Loaded {len(full_dataset)} training samples")

    val_size = int(len(full_dataset) * config.val_split)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(full_dataset), generator=generator)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)

    val_dataset_base = PMCVQADataset(
        csv_path=config.train_csv,
        image_dir=config.image_dir,
        tokenizer=tokenizer,
        max_length=config.max_text_length,
        image_size=config.image_size,
        split='train',
        train=False,
    )
    if limit and limit < len(val_dataset_base):
        val_dataset_base.samples = val_dataset_base.samples[:limit]
    val_dataset = torch.utils.data.Subset(val_dataset_base, val_indices)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    class_weights = compute_class_weights_from_csv(
        config.train_csv, num_samples=limit or 0)
    print(f"Class weights: {class_weights.tolist()}")

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
        collate_fn=collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_size=config.eval_batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
        collate_fn=collate_fn)

    model = PMCVQAModel(config).to(device)
    trainable = count_trainable_params(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}  |  Trainable: {trainable:,}  "
          f"({100 * trainable / total:.1f}%)")

    head_params = get_fusion_head_params(model)
    lora_params = get_lora_params(model)

    if lora_params:
        optimizer = AdamW([
            {'params': lora_params, 'lr': config.lr},
            {'params': head_params, 'lr': config.head_lr},
        ], weight_decay=config.weight_decay)
    else:
        optimizer = AdamW(head_params, lr=config.head_lr,
                          weight_decay=config.weight_decay)

    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    def warmup_cosine_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_lambda)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=config.label_smoothing,
    )
    scaler = GradScaler('cuda', enabled=(config.use_amp and device.type == 'cuda'))

    best_val_acc = 0.0
    best_metrics = {}
    epochs_no_improve = 0
    for epoch in range(1, config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.num_epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler,
            device, config)

        (val_loss, val_acc, per_class_acc,
         per_class_precision, per_class_recall, per_class_f1,
         macro_f1, top2_acc, auc_roc, val_labels_np, val_preds_np,
         confusion_np) = validate(model, val_loader, criterion, device)

        labels = ['A', 'B', 'C', 'D']
        per_class_str = ', '.join(
            f"{l}: {a:.4f}" for l, a in zip(labels, per_class_acc))
        f1_str = ', '.join(
            f"{l}: {f:.4f}" for l, f in zip(labels, per_class_f1))

        print(f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")
        print(f"  Per-class Acc: {per_class_str}")
        print(f"  Per-class F1 : {f1_str}  |  Macro F1: {macro_f1:.4f}  "
              f"Top-2: {top2_acc:.4f}  AUC-ROC: {auc_roc:.4f}")

        if config.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/acc': train_acc,
                'val/loss': val_loss,
                'val/acc': val_acc,
                'val/per_class_A': per_class_acc[0],
                'val/per_class_B': per_class_acc[1],
                'val/per_class_C': per_class_acc[2],
                'val/per_class_D': per_class_acc[3],
                'val/precision_A': per_class_precision[0],
                'val/precision_B': per_class_precision[1],
                'val/precision_C': per_class_precision[2],
                'val/precision_D': per_class_precision[3],
                'val/recall_A': per_class_recall[0],
                'val/recall_B': per_class_recall[1],
                'val/recall_C': per_class_recall[2],
                'val/recall_D': per_class_recall[3],
                'val/f1_A': per_class_f1[0],
                'val/f1_B': per_class_f1[1],
                'val/f1_C': per_class_f1[2],
                'val/f1_D': per_class_f1[3],
                'val/macro_f1': macro_f1,
                'val/top2_acc': top2_acc,
                'val/auc_roc': auc_roc,
                'val/confusion_matrix': wandb.plot.confusion_matrix(
                    y_true=val_labels_np, preds=val_preds_np,
                    class_names=labels),
                'lr': scheduler.get_last_lr()[0],
            })

        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'train_acc': train_acc,
            'config': config,
        }
        torch.save(ckpt, f"{config.checkpoint_dir}/last.pt")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = {
                'best_val_acc': val_acc,
                'best_macro_f1': macro_f1,
                'best_top2_acc': top2_acc,
                'best_auc_roc': auc_roc,
                'best_f1_A': per_class_f1[0],
                'best_f1_B': per_class_f1[1],
                'best_f1_C': per_class_f1[2],
                'best_f1_D': per_class_f1[3],
                'best_val_loss': val_loss,
            }
            epochs_no_improve = 0
            torch.save(ckpt, f"{config.checkpoint_dir}/best.pt")
            print(f"  saved best model (val_acc={val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.early_stopping_patience:
                print(f"  early stopping after {epoch} epochs (no improvement for "
                      f"{config.early_stopping_patience} epochs)")
                break

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")

    if config.use_wandb:
        for k, v in best_metrics.items():
            wandb.run.summary[k] = v
        ckpt = torch.load(f"{config.checkpoint_dir}/best.pt", map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"\nGenerating explanations for {config.num_explain_samples} samples...")
        explain_samples(model, val_dataset_base, tokenizer, device, config,
                        num_samples=config.num_explain_samples)
        explain_dir = config.explain_dir
        if os.path.isdir(explain_dir):
            images = []
            for fname in sorted(os.listdir(explain_dir)):
                if fname.endswith('.png'):
                    path = os.path.join(explain_dir, fname)
                    images.append(wandb.Image(path, caption=fname))
            if images:
                wandb.log({"explanations": images})
                print(f"  Logged {len(images)} explanation images to W&B")

    if config.push_to_hub and config.hf_repo_id:
        print(f"Pushing best model to Hugging Face Hub: {config.hf_repo_id}")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=f"{config.checkpoint_dir}/best.pt",
            path_in_repo="model.pt",
            repo_id=config.hf_repo_id,
            repo_type="model",
        )
        print(f"  Model pushed to https://huggingface.co/{config.hf_repo_id}")

    if config.use_wandb:
        wandb.finish()

    with open(f"{config.output_dir}/train_results.json", 'w') as f:
        json.dump({
            **best_metrics,
            'config': {k: (str(v) if not isinstance(v, (int, float, bool, str))
                           else v)
                       for k, v in config.__dict__.items()},
        }, f, indent=2)


if __name__ == '__main__':
    main()
