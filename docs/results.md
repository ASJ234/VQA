# Training Results — PMC-VQA

## Metrics and why they were chosen

### Accuracy
Overall fraction of correct predictions. Simple and intuitive, but misleading when classes are imbalanced — a model guessing "C" every time would get ~38%.

### Per-class accuracy
Accuracy broken down per class (A/B/C/D). Reveals which classes the model handles well or poorly — critical for spotting imbalance effects.

### Precision (per-class)
Of all predictions for class X, how many were correct?
```
precision = tp / (tp + fp)
```
High precision = few false alarms. Important when a wrong answer is costly (e.g., confident but wrong diagnosis).

### Recall (per-class)
Of all true class X samples, how many did the model catch?
```
recall = tp / (tp + fn)
```
High recall = few misses. Important when missing the correct answer is costly.

### F1 (per-class)
Harmonic mean of precision and recall:
```
F1 = 2 · P · R / (P + R)
```
A single number balancing both — better than accuracy when classes are imbalanced.

### Macro F1
Unweighted average of per-class F1 scores. Treats all classes equally regardless of size — the **single best metric** for imbalanced datasets. If the model ignores minority classes (A, D), macro F1 drops sharply.

### Top-2 accuracy
Is the correct answer among the model's top 2 choices? At 67% vs 42% accuracy, it shows the model is usually "warm" — it narrows it down to two options even when it can't pick the right one. Useful for applications with human-in-the-loop verification.

### AUC-ROC (One-vs-Rest)
Measures how well the model separates each class from all others across all confidence thresholds. 0.5 = random, 1.0 = perfect. At 0.67, the model clearly ranks correct answers above incorrect ones on average, even when it doesn't always pick the right one.

### Confusion matrix
4×4 grid showing true vs predicted labels. Reveals specific confusion patterns — e.g., is the model consistently mixing up A↔D (both minority classes) or B↔C (both majority classes)?

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

| Metric | Value |
|---|---|
| Best val accuracy | 41.8% |
| Macro F1 | 0.402 |
| Top-2 accuracy | 67.2% |
| AUC-ROC (OVR) | 0.670 |
| Best epoch | 11 (early stopped at 15) |

### Per-class metrics (at best epoch)

| Class | Acc | Precision | Recall | F1 |
|---|---|---|---|---|
| A | 44.7% | 0.281 | 0.447 | 0.345 |
| B | 44.0% | 0.471 | 0.440 | 0.455 |
| C | 38.9% | 0.569 | 0.389 | 0.463 |
| D | 41.2% | 0.298 | 0.412 | 0.346 |

## Analysis

- **Val loss stabilized** after epoch 11 (didn't climb back up) — overfitting fix worked
- **AUC-ROC 0.67** and **Top-2 67%** show the model learns meaningful representations, well above random (50% / 25%)
- **Class imbalance** hurts minority classes (A ~14%, D ~13%): lower precision/F1 vs majority classes (B ~36%, C ~38%)
- **Accuracy plateaued at ~42%** — the model can't reliably pick the single correct answer out of 4

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
