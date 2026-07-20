import os
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm

from config import Config
from dataset import PMCVQADataset, collate_fn
from model import PMCVQAModel


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

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
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean()
    per_class_acc = {}
    for c in range(4):
        mask = all_labels == c
        per_class_acc[chr(65 + c)] = float((all_preds[mask] == all_labels[mask]).mean()) if mask.sum() > 0 else 0.0

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds,
                                   target_names=['A', 'B', 'C', 'D'],
                                   output_dict=True)

    return total_loss / len(loader.dataset), acc, per_class_acc, cm.tolist(), report


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
    loss, acc, per_class_acc, cm, report = evaluate(
        model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"  Loss: {loss:.4f}  |  Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  Per-class accuracy: {per_class_acc}")
    print(f"\nConfusion Matrix:")
    for row in cm:
        print(f"    {row}")
    print(f"\nPer-class Metrics:")
    for cls_name in ['A', 'B', 'C', 'D']:
        if cls_name in report:
            r = report[cls_name]
            print(f"  {cls_name}: precision={r['precision']:.4f}, "
                  f"recall={r['recall']:.4f}, f1={r['f1-score']:.4f}")

    os.makedirs(config.output_dir, exist_ok=True)
    results = {
        'loss': loss,
        'accuracy': acc,
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': cm,
        'classification_report': report,
    }
    with open(f"{config.output_dir}/test_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {config.output_dir}/test_results.json")

    if args.explain:
        from explain import explain_samples
        explain_samples(model, test_dataset, tokenizer, device, config,
                        num_samples=args.num_explain)


if __name__ == '__main__':
    main()
