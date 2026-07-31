# Changelog — 2026-07-31

Changes from the metrics-overhaul session.

---

## 1. Metric set replaced with Accuracy, WUPS, BLEU, F1-Score

Added `src/metrics.py` with a single `compute_all_metrics()` entry point. The model is now evaluated with exactly four metrics:

| Metric | Description |
|---|---|
| **Accuracy** | Fraction of correct predictions (predicted index == ground-truth index) |
| **WUPS** | Wu-Palmer (WordNet) semantic similarity between predicted and reference answer text; reported at thresholds 0.0 and 0.9 |
| **BLEU** | n-gram precision + brevity penalty between predicted and reference answer text; reported as BLEU-1 … BLEU-4 |
| **F1-Score** | Macro F1 (unweighted mean of per-class F1) plus per-class F1 (A/B/C/D) |

Because the task is multiple-choice, WUPS and BLEU compare the **predicted choice text** against the **reference choice text** of the correct option.

### Removed metrics

| Removed | Reason |
|---|---|
| Per-class accuracy | Replaced by per-class F1 |
| Per-class precision / recall | Folded into F1 |
| Top-2 accuracy | Not part of the requested metric set |
| AUC-ROC (OVR) | Not part of the requested metric set |
| Confusion matrix | Not part of the requested metric set |

### `src/metrics.py` (new)
- `compute_accuracy`, `compute_f1`, `compute_bleu`, `compute_wups` (self-contained BLEU, WordNet-based WUPS via NLTK)
- `compute_all_metrics(labels, preds, pred_texts, ref_texts)` → dict with all four metrics

### `src/dataset.py`
- `__getitem__` now also returns `choices` (the four cleaned choice texts) so eval/train can build predicted/reference answer strings for WUPS & BLEU.

### `src/eval.py`
- `evaluate()` now returns `{loss, accuracy, f1_macro, f1, wups_0.0, wups_0.9, bleu}`.
- Removed confusion matrix, classification report, per-class accuracy printing.
- Results saved to `outputs/test_results.json`.

### `src/train.py`
- `validate()` now returns the same four-metric dict.
- Console output, W&B per-epoch logging (`val/acc`, `val/macro_f1`, `val/f1_A…D`, `val/wups_0.0`, `val/wups_0.9`, `val/bleu_1…4`) and `wandb.run.summary` best-metrics updated to the new set.
- Best checkpoint selection still uses validation accuracy.

### `requirements.txt`
- Added `nltk` (WordNet data downloaded automatically on first WUPS use).

---

## Commits

```
31e3b83  use Accuracy, WUPS, BLEU, F1-Score metrics for VQA evaluation
```

---

# Changelog — 2026-07-28

All changes from the overfitting-fix session.

---

## 1. Anti-overfitting measures

### `src/config.py`

| Setting | Before | After | Effect |
|---|---|---|---|
| `fusion_dropout` | 0.3 | **0.5** | More aggressive dropout in the classification head |
| `head_lr` | 1e-3 | **5e-4** | Lower learning rate for the head (fewer params, less prone to overfit) |
| `weight_decay` | 0.01 | **0.1** | Stronger L2 regularization |
| `label_smoothing` | — | **0.1** | Softens one-hot targets, prevents overconfidence |
| `early_stopping_patience` | — | **4** | Stops when val acc fails to improve for 4 consecutive epochs |

### `src/train.py`

- **Label smoothing loss**: `CrossEntropyLoss` now uses `label_smoothing=config.label_smoothing`
- **Early stopping**: tracks epochs without val acc improvement; breaks and keeps best checkpoint when patience is exceeded

---

## 2. Extended metrics

`src/train.py` — `validate()` now collects all predictions and probabilities per epoch and computes:

| Metric | Description |
|---|---|
| **Per-class accuracy** | Accuracy per class (A/B/C/D) |
| **Per-class precision** | `tp / (tp + fp)` per class |
| **Per-class recall** | `tp / (tp + fn)` per class |
| **Per-class F1** | `2 * P * R / (P + R)` per class |
| **Macro F1** | Unweighted mean of per-class F1 |
| **Top-2 accuracy** | Is the correct answer in the top-2 predictions? |
| **AUC-ROC (OVR)** | One-vs-rest area under the ROC curve |
| **Confusion matrix** | 4×4 matrix of true vs predicted labels |

All logged to W&B as per-epoch charts, plus `val/confusion_matrix` as an interactive plot.

---

## 3. W&B summary

At the end of training, `wandb.run.summary` is set with all best metrics:
`best_val_acc`, `best_macro_f1`, `best_top2_acc`, `best_auc_roc`, `best_f1_A`–`best_f1_D`, `best_val_loss`.

These appear on the W&B run overview page.

---

## 4. XAI explanations in W&B

After training completes, the best checkpoint is loaded and `explain_samples()` runs on the validation set. ViT attention heatmaps + gradient saliency figures are logged as `wandb.Image` under the `"explanations"` key.

---

## 5. Optimised Hugging Face Hub push

Previously pushed `best.pt` (full checkpoint ~818 MB including optimizer state).

Now pushes only:

| File | Size | Contents |
|---|---|---|
| `adapter_model.pt` | ~11 MB | State dict of trainable params only (LoRA A/B matrices + fusion head) |
| `config.json` | ~1 KB | Hyperparameters to rebuild the model |
| `train_results.json` | ~1 KB | Best evaluation metrics |

Backbone (BiomedCLIP, 196M params) stays on its original HF repo — no duplicate upload.

### Loading the adapter

```python
from src.model import PMCVQAModel
from src.config import Config

config = Config()
model = PMCVQAModel(config)
model.load_state_dict(torch.load("adapter_model.pt"), strict=False)
```

---

## 6. Unused imports

Removed `urllib.request` from `download_data.py` (unused).

---

## Commits

```
d13290a  remove unused import urllib.request
bb5914b  push only trainable adapter weights (LoRA + head) to HF Hub
3d4a22b  log all best metrics to wandb summary and results json
4f90871  add precision, recall, F1, top-2, AUC-ROC, confusion matrix to wandb
49312c3  log best_val_acc to wandb summary
4bdab6f  log XAI explanation figures to W&B after training
78d6203  push lightweight model (no optimizer), config.json, and metrics to HF Hub
f51d2b1  fix overfitting: stronger reg, label smoothing, early stopping
```
