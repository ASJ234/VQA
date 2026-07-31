# Training Results — PMC-VQA

## Metrics and why they were chosen

The model is evaluated with four metrics: **Accuracy, WUPS, BLEU, and F1-Score**. Because this is a multiple-choice VQA task (pick one of A/B/C/D), the text-based metrics (WUPS, BLEU) are computed by comparing the **predicted choice text** against the **reference choice text** of the correct answer.

### Accuracy
Overall fraction of correct predictions (`predicted index == ground-truth index`). Simple and intuitive, but misleading when classes are imbalanced — a model guessing "C" every time would get ~38%.

### WUPS (Wu-Palmer Similarity)
Semantic similarity between the predicted answer text and the reference answer text, computed via WordNet. It measures how "close" a wrong answer is to the right one:

- **WUPS@0.0** — plain mean Wu-Palmer similarity (0 = unrelated, 1 = identical)
- **WUPS@0.9** — stricter: similarities below 0.9 are penalized by scaling with 0.9

A partial-credit metric — the model gets some credit for answers that are semantically similar (e.g. "Red" vs "Reddish") even when not exact.

### BLEU
n-gram precision between the predicted answer text and the reference answer text with a brevity penalty. Reported as BLEU-1 through BLEU-4. Measures exact word/sequence overlap rather than semantics.

### F1-Score
Harmonic mean of precision and recall. Reported as **macro F1** (unweighted mean of per-class F1), the single best metric for imbalanced datasets — if the model ignores minority classes (A, D), macro F1 drops sharply. Per-class F1 (A/B/C/D) is also reported.

All four metrics are logged to W&B per epoch during training, written to the run summary at the end of training, and saved to `outputs/test_results.json` after evaluation.

## What is LoRA?

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique. Instead of updating a full weight matrix W (e.g. 768×768), it injects two tiny matrices A and B alongside it:

```
W_updated = W + B·A
           ↑ frozen   ↑ trainable (rank r)
```

- **A** is `(r × in_dim)`, **B** is `(out_dim × r)` — very small
- Only A and B are trained; the original W stays frozen
- This preserves the backbone's pretrained knowledge while adapting to the new task
- **Result**: only ~2.9M trainable params (1.4% of 196M), enabling fine-tuning on a single GPU with limited data, without catastrophic forgetting or overfitting

## Summary

*Pending — a fresh run is needed to populate the table with the new metric set. Values below are from the previous run under the old metric set.*

| Metric | Value |
|---|---|
| Best val accuracy | 41.8% |
| Macro F1 | 0.402 |
| Top-2 accuracy | 67.2% |
| AUC-ROC (OVR) | 0.670 |
| Best epoch | 11 (early stopped at 15) |

> Note: Top-2 accuracy and AUC-ROC were removed when the metric set was replaced. They are kept here only as a historical reference for the previous run.

## Analysis

- **Val loss stabilized** after epoch 11 (didn't climb back up) — overfitting fix worked
- **Class imbalance** hurts minority classes (A ~14%, D ~13%): lower F1 vs majority classes (B ~36%, C ~38%)
- **Accuracy plateaued at ~42%** — the model can't reliably pick the single correct answer out of 4
- The new **WUPS and BLEU** metrics will reveal how semantically close the model's wrong answers are to the ground truth

## How to improve

### 1. Increase LoRA capacity

| Setting | Current | Suggested |
|---|---|---|
| `lora_r` | 8 | 16–32 |
| `lora_alpha` | 16 | 32–64 |
| `lora_dropout` | 0.1 | 0.05 |

More LoRA parameters let the backbone adapt better to medical domain features.

### 2. Bigger fusion head

Increase `fusion_hidden` from 512 → 1024, or add a second hidden layer. The current head is Linear(1536 → 512 → 1) — may be too small to model complex cross-modal interactions.

### 3. More training data

9500 samples is modest for medical VQA. Sources:
- Use the full PMC-VQA dataset (not limited to 10k)
- Add data augmentation: RandAugment, CutMix, or medical-specific transforms (color jitter, elastic deformation)
- Generate synthetic QA pairs from figure captions

### 4. Full backbone fine-tuning

Unfreeze the vision encoder (or last K layers) after initial LoRA convergence:
1. First train LoRA + head for 10 epochs
2. Then unfreeze vision backbone with a lower LR (1e-6 – 5e-6) for 5 more epochs

### 5. Ensemble / multi-scale

- Ensemble 3–5 models trained with different seeds
- Use multi-crop inference (test-time augmentation)

### 6. Class imbalance handling

Current class weights help but could be stronger — try focal loss instead of weighted cross-entropy:

```python
loss = FocalLoss(gamma=2.0, weight=class_weights)
```

### 7. Longer training

With the anti-overfitting measures in place, increase `num_epochs` to 25–30 and let early stopping decide when to stop.

## Next logical steps (highest impact)

1. Increase `lora_r` to 16 and `fusion_hidden` to 1024
2. Train on 20k–30k samples (remove `max_train_samples`)
3. If still plateaued < 50%, try unfreezing last 6 ViT blocks
