# Training Results — PMC-VQA

## Metrics and why they were chosen

The model is evaluated with four metrics: **Accuracy, WUPS, BLEU, and F1-Score**. Because this is a multiple-choice VQA task (pick one of A/B/C/D), the text-based metrics (WUPS, BLEU) are computed by comparing the **predicted choice text** against the **reference choice text** of the correct answer.

### Accuracy
Overall fraction of correct predictions (`predicted index == ground-truth index`). Simple and intuitive, but misleading when classes are imbalanced — a model guessing "C" every time would get ~38%.

**Importance:** the headline number — "how often does the model pick the exact correct option (A/B/C/D)?" Read it together with F1, since imbalance can inflate it.

### WUPS (Wu-Palmer Similarity)
Semantic similarity between the predicted answer text and the reference answer text, computed via WordNet. It measures how "close" a wrong answer is to the right one:

- **WUPS@0.0** — plain mean Wu-Palmer similarity (0 = unrelated, 1 = identical)
- **WUPS@0.9** — stricter: similarities below 0.9 are penalized by scaling with 0.9

A partial-credit metric — the model gets some credit for answers that are semantically similar (e.g. "Red" vs "Reddish") even when not exact.

**Importance:** reveals *how semantically close the wrong answers are to the truth*. High WUPS with moderate accuracy means errors are near-misses; low WUPS means the model is essentially guessing off-base answers.

### BLEU
n-gram precision between the predicted answer text and the reference answer text with a brevity penalty. Reported as BLEU-1 through BLEU-4. Measures exact word/sequence overlap rather than semantics.

**Importance:** complements WUPS — WUPS measures *meaning*, BLEU measures *surface text match*. Together they show whether the model's generated answer text is of meaningful quality, not just that the chosen letter happens to be right.

### F1-Score
Harmonic mean of precision and recall. Reported as **macro F1** (unweighted mean of per-class F1), the single best metric for imbalanced datasets — if the model ignores minority classes (A, D), macro F1 drops sharply. Per-class F1 (A/B/C/D) is also reported.

**Importance:** the honesty check on accuracy — "is performance balanced across all four options?" A macro F1 noticeably below accuracy signals the model is neglecting minority classes.

### Summary of metric roles

| Metric | Answers | Role |
|---|---|---|
| Accuracy | "How often is the exact option right?" | Choice selection |
| F1-Score | "Are all classes handled fairly?" | Choice selection |
| WUPS | "How semantically close are wrong answers?" | Answer text quality |
| BLEU | "How much text overlap with the reference?" | Answer text quality |

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

Run logged at **epoch 10** (best checkpoint).

| Metric | Value |
|---|---|
| Best val accuracy | 41.2% |
| Val macro F1 | 0.378 |
| Best macro F1 | 0.403 |
| Per-class F1 (best) | A: 0.347 · B: 0.446 · C: 0.449 · D: 0.372 |
| WUPS@0.0 (best) | 0.832 |
| WUPS@0.9 (best) | 0.804 |
| BLEU-4 (best) | 0.583 |
| Best val loss | 1.426 |
| Best epoch | 10 |

**Test set:**

| Metric | Value |
|---|---|
| Accuracy | 37.2% |
| Macro F1 | 0.357 |
| WUPS@0.0 | 0.811 |
| WUPS@0.9 | 0.781 |
| BLEU-1 / -2 / -3 / -4 | 0.627 / 0.586 / 0.566 / 0.556 |

> Note: Top-2 accuracy and AUC-ROC were removed when the metric set was replaced. The previous run reported Top-2 accuracy of 67.2% and AUC-ROC (OVR) of 0.670; kept here only as a historical reference.

## Analysis

- **Val loss stabilized** around epoch 10 (best val loss 1.426) — overfitting fix worked
- **Class imbalance** hurts minority classes (A ~14%, D ~13%): lower F1 vs majority classes (B ~36%, C ~38%)
- **Accuracy plateaued at ~41%** — the model can't reliably pick the single correct answer out of 4
- **WUPS and BLEU** show the model's wrong answers are still semantically close to the ground truth (WUPS@0.9 ≈ 0.80, BLEU-4 ≈ 0.58), i.e. errors are near-misses rather than off-base guesses

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
