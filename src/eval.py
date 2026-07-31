import os
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from config import Config
from dataset import PMCVQADataset, collate_fn
from model import PMCVQAModel
from metrics import compute_all_metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    pred_texts = []
    ref_texts = []

    for batch in tqdm(loader, desc='Eval'):
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
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        all_preds.extend(preds_np)
        all_labels.extend(labels_np)

        choices = batch['choices']
        pred_texts.extend([choices[i][int(p)] for i, p in enumerate(preds_np)])
        ref_texts.extend([choices[i][int(l)] for i, l in enumerate(labels_np)])

    metrics = compute_all_metrics(all_labels, all_preds, pred_texts, ref_texts)
    metrics['loss'] = total_loss / len(loader.dataset)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt')
    parser.add_argument('--explain', action='store_true')
    parser.add_argument('--num_explain', type=int, default=10)
    args = parser.parse_args()

    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model_config = ckpt.get('config', config)
    for k, v in config.__dict__.items():
        if hasattr(model_config, k):
            setattr(config, k, getattr(model_config, k))

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True)

    test_dataset = PMCVQADataset(
        csv_path=config.test_csv,
        image_dir=config.image_dir,
        tokenizer=tokenizer,
        max_length=config.max_text_length,
        image_size=config.image_size,
        split='test',
    )
    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset, batch_size=config.eval_batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
        collate_fn=collate_fn)

    model = PMCVQAModel(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"  Loss:    {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"  Macro F1: {metrics['f1_macro']:.4f}")
    for k, v in metrics['f1'].items():
        print(f"    F1 {k}: {v:.4f}")
    print(f"  WUPS@0.0: {metrics['wups_0.0']:.4f}")
    print(f"  WUPS@0.9: {metrics['wups_0.9']:.4f}")
    for k, v in metrics['bleu'].items():
        print(f"  {k.upper()}: {v:.4f}")

    os.makedirs(config.output_dir, exist_ok=True)
    results = {k: v for k, v in metrics.items()}
    with open(f"{config.output_dir}/test_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {config.output_dir}/test_results.json")

    if args.explain:
        from explain import explain_samples
        explain_samples(model, test_dataset, tokenizer, device, config,
                        num_samples=args.num_explain)


if __name__ == '__main__':
    main()
