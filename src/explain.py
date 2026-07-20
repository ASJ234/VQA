import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def _get_vit_attention(model, image_tensor, device):
    attentions = []

    def hook_fn(module, args, output):
        if isinstance(output, tuple) and len(output) > 1:
            attentions.append(output[1].detach())

    hooks = []
    if hasattr(model.backbone.vision_encoder, 'transformer'):
        for block in model.backbone.vision_encoder.transformer.resblocks:
            if hasattr(block, 'attn'):
                hook = block.attn.register_forward_hook(hook_fn)
                hooks.append(hook)

    model.eval()
    with torch.no_grad():
        _ = model.backbone.vision_encoder(image_tensor.unsqueeze(0).to(device))

    for h in hooks:
        h.remove()

    if not attentions:
        return None

    attn_maps = [a[0].mean(0).cpu().numpy() for a in attentions]

    seq_len = attn_maps[0].shape[0]
    I = np.eye(seq_len)
    processed = []
    for att in attn_maps:
        att = (att + I) / 2
        att = att / att.sum(axis=1, keepdims=True)
        processed.append(att)

    rollout = processed[0]
    for att in processed[1:]:
        rollout = rollout @ att

    cls_attn = rollout[0, 1:]
    num_patches = cls_attn.shape[0]
    h = w = int(np.sqrt(num_patches))
    return cls_attn.reshape(h, w)


def _get_gradient_saliency(model, images, q_ids, q_mask, c_ids, c_mask, pred_label, device):
    images = images.clone().detach().requires_grad_(True).to(device)
    q_ids = q_ids.to(device)
    q_mask = q_mask.to(device)
    c_ids = c_ids.to(device)
    c_mask = c_mask.to(device)

    model.eval()
    scores = model(images, q_ids, q_mask, c_ids, c_mask)
    score = scores[0, pred_label]

    model.zero_grad()
    score.backward()

    grad = images.grad[0].cpu().numpy()
    saliency = np.max(np.abs(grad), axis=0)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency


def _format_text(tokenizer, ids):
    tokens = tokenizer.convert_ids_to_tokens(ids.cpu().numpy().tolist())
    tokens = [t for t in tokens if t not in ('[PAD]', '<pad>', '[SEP]', '<|endoftext|>', '[CLS]')]
    text = ''.join(t.replace('##', '').replace('Ġ', ' ') for t in tokens).strip()
    return text


def explain_samples(model, dataset, tokenizer, device, config, num_samples=10):
    os.makedirs(config.explain_dir, exist_ok=True)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for idx in indices:
        sample = dataset[idx]
        img_path = dataset.image_dir + '/' + dataset.samples[idx]['Figure_path']
        true_label = chr(65 + sample['label'].item())

        model.eval()
        with torch.no_grad():
            scores = model(
                sample['image'].unsqueeze(0).to(device),
                sample['question_input_ids'].unsqueeze(0).to(device),
                sample['question_attention_mask'].unsqueeze(0).to(device),
                sample['choices_input_ids'].unsqueeze(0).to(device),
                sample['choices_attention_mask'].unsqueeze(0).to(device),
            )
        pred_label = scores.argmax(dim=-1).item()
        pred_label_char = chr(65 + pred_label)

        heatmap = _get_vit_attention(model, sample['image'], device)
        saliency = _get_gradient_saliency(
            model,
            sample['image'].unsqueeze(0).to(device),
            sample['question_input_ids'].unsqueeze(0).to(device),
            sample['question_attention_mask'].unsqueeze(0).to(device),
            sample['choices_input_ids'].unsqueeze(0).to(device),
            sample['choices_attention_mask'].unsqueeze(0).to(device),
            pred_label, device)

        _save_explanation_figure(
            img_path, heatmap, saliency,
            sample['question_input_ids'], sample['choices_input_ids'],
            tokenizer, pred_label_char, true_label,
            scores.softmax(dim=-1).squeeze().cpu().numpy().tolist(),
            f"{config.explain_dir}/{dataset.samples[idx]['Figure_path']}.png")

        print(f"  Explained {dataset.samples[idx]['Figure_path']}: "
              f"pred={pred_label_char} true={true_label}")


def _save_explanation_figure(img_path, heatmap, saliency, q_ids, c_ids,
                              tokenizer, pred_label, true_label, probs, save_path):
    img = np.array(Image.open(img_path).convert('RGB'))
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Image", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    if heatmap is not None:
        hm = np.array(Image.fromarray(heatmap).resize(
            (img.shape[1], img.shape[0]), Image.BILINEAR))
        axes[0, 1].imshow(img, alpha=0.6)
        im = axes[0, 1].imshow(hm, alpha=0.4, cmap='jet')
        axes[0, 1].set_title("ViT Attention Heatmap", fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    else:
        axes[0, 1].text(0.5, 0.5, 'Attention\nnot available', transform=axes[0, 1].transAxes,
                        ha='center', va='center')
        axes[0, 1].axis('off')

    if saliency is not None:
        s = np.array(Image.fromarray(saliency).resize(
            (img.shape[1], img.shape[0]), Image.BILINEAR))
        axes[0, 2].imshow(img, alpha=0.6)
        axes[0, 2].imshow(s, alpha=0.4, cmap='hot')
        axes[0, 2].set_title("Gradient Saliency", fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
    else:
        axes[0, 2].text(0.5, 0.5, 'Saliency\nnot available', transform=axes[0, 2].transAxes,
                        ha='center', va='center')
        axes[0, 2].axis('off')

    q_text = _format_text(tokenizer, q_ids)
    axes[1, 0].axis('off')
    axes[1, 0].set_title("Question", fontsize=12, fontweight='bold')
    axes[1, 0].text(0.05, 0.5, q_text, fontsize=10, wrap=True,
                    transform=axes[1, 0].transAxes, verticalalignment='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', alpha=0.9))

    choices_text = [_format_text(tokenizer, c_ids[i]) for i in range(4)]
    ax = axes[1, 1]
    ax.axis('off')
    ax.set_title("Choices & Probabilities", fontsize=12, fontweight='bold')
    for i in range(4):
        label = chr(65 + i)
        is_gt = label == true_label
        is_pred = label == pred_label
        color = '#d4edda' if is_gt else '#f8d7da' if is_pred and not is_gt else '#fff'
        marker = '✓' if is_gt else ('✗' if is_pred else '')
        ax.text(0.05, 0.85 - i * 0.22,
                f"{label}: {choices_text[i][:50]}{'...' if len(choices_text[i]) > 50 else ''}",
                fontsize=10, color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
        ax.text(0.75, 0.85 - i * 0.22,
                f"{probs[i]:.3f} {marker}",
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))

    summary_ax = axes[1, 2]
    summary_ax.axis('off')
    summary_ax.set_title("Summary", fontsize=12, fontweight='bold')
    summary_text = (
        f"Ground Truth: {true_label}\n"
        f"Prediction:   {pred_label}\n"
        f"Correct: {'✓' if pred_label == true_label else '✗'}\n\n"
        f"Choice Probabilities:\n"
        + '\n'.join(f"  {chr(65+i)}: {p:.3f}" for i, p in enumerate(probs))
    )
    summary_ax.text(0.05, 0.5, summary_text, fontsize=11, verticalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
