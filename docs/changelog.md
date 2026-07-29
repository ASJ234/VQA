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
