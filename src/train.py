import os
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
from tqdm import tqdm

from config import Config
from dataset import PMCVQADataset, collate_fn
from model import (PMCVQAModel, get_fusion_head_params, get_lora_params,
                   count_trainable_params)


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
            with autocast():
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
    correct = 0
    total = 0
    per_class_correct = torch.zeros(4, device=device)
    per_class_total = torch.zeros(4, device=device)

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
        preds = scores.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        for c in range(4):
            mask = labels == c
            per_class_correct[c] += (preds[mask] == labels[mask]).sum().item()
            per_class_total[c] += mask.sum().item()

    acc = correct / total
    per_class_acc = per_class_correct / per_class_total.clamp(min=1)
    return total_loss / total, acc, per_class_acc.tolist()


def main():
    config = Config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True)

    full_dataset = PMCVQADataset(
        csv_path=config.train_csv,
        image_dir=config.image_dir,
        tokenizer=tokenizer,
        max_length=config.max_text_length,
        image_size=config.image_size,
        split='train',
    )

    limit = config.max_train_samples
    if limit and limit < len(full_dataset):
        full_dataset.samples = full_dataset.samples[:limit]
    print(f"Loaded {len(full_dataset)} training samples")

    val_size = int(len(full_dataset) * config.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42))
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

    def warmup_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    scaler = GradScaler(enabled=(config.use_amp and device.type == 'cuda'))

    best_val_acc = 0.0
    for epoch in range(1, config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.num_epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler,
            device, config)

        val_loss, val_acc, per_class_acc = validate(
            model, val_loader, criterion, device)

        labels = ['A', 'B', 'C', 'D']
        per_class_str = ', '.join(
            f"{l}: {a:.4f}" for l, a in zip(labels, per_class_acc))

        print(f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")
        print(f"  Per-class: {per_class_str}")

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'train_acc': train_acc,
            'config': config,
        }
        torch.save(ckpt, f"{config.checkpoint_dir}/last.pt")
        if is_best:
            torch.save(ckpt, f"{config.checkpoint_dir}/best.pt")
            print(f"  saved best model (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")

    with open(f"{config.output_dir}/train_results.json", 'w') as f:
        json.dump({
            'best_val_acc': best_val_acc,
            'config': {k: (str(v) if not isinstance(v, (int, float, bool, str))
                           else v)
                       for k, v in config.__dict__.items()},
        }, f, indent=2)


if __name__ == '__main__':
    main()
