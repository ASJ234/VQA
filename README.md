# PMC-VQA: Medical Visual Question Answering

Visual Question Answering model for the [PMC-VQA](https://github.com/xiaoyi12/PMC-VQA) dataset, fine-tuned with **BiomedCLIP** + **LoRA** and enhanced with explainable AI (XAI) visualizations.

---

## Dataset EDA

Run `python src/eda.py` to generate all analysis plots and summary statistics under `outputs/eda/`.

### Key Findings (Version 2)

| Metric | Value |
|--------|-------|
| Training samples | 152,603 |
| Test samples | 33,430 |
| Unique images | 164,360 |
| Train/Test image overlap | **0** (clean split) |
| Avg image size | 290 × 264 px |
| Avg question length | 9.4 words |
| Avg caption length | 26.8 words |
| Questions per image | 86.8% have 1 question |

### Label Distribution (V2, skewed → weighted loss used)

```
A: 13.78%   B: 35.61%   C: 37.80%   D: 12.81%
```

### Question Types

77.1% start with "What", 15.3% with "Which", 4.6% with "Where".

### Generated Plots

| Plot | Description |
|------|-------------|
| `dataset_sizes.png` | Sample counts per split |
| `label_distribution.png` | Answer label distribution (V2 train & test) |
| `question_types.png` | Question starting word histogram |
| `image_statistics.png` | Width, height, aspect ratio, file size |
| `text_lengths.png` | Word count distributions for Q, choices, captions |
| `questions_per_image.png` | How many QA pairs per image |
| `train_test_overlap.png` | Train/test image overlap (zero) |
| `eda_summary.json` | All metrics in machine-readable format |

---

## Dataset

| Split | Version 1 | Version 2 (noncompound, used here) |
|-------|-----------|-----------------------------------|
| Train | ~177k QA pairs | ~152k QA pairs |
| Test  | ~50k QA pairs  | ~35k QA pairs  |
| Test Clean | 2k QA pairs | — |
| Images | ~149k | ~164k (figures) |

Each sample contains: `Figure_path`, `Caption`, `Question`, `Choice A/B/C/D`, `Answer`.

---

## Architecture

```
                    ┌─────────────────────┐
Image ────► ViT-B/16 ──► img_feat (768) ──┐
                    │                     │
Question ──► PubMedBERT ──► q_feat (768) ─┼──► [img; q; choice_i] (2304)
                    │                     │       → Linear → GELU → Dropout → Linear → 1
Choice A ──► PubMedBERT ──► cA_feat──────┘       → repeat for B, C, D
Choice B ──► PubMedBERT ──► cB_feat──────┐       → stack 4 scores → CrossEntropy
Choice C ──► PubMedBERT ──► cC_feat──────┘
Choice D ──► PubMedBERT ──► cD_feat──────┘
```

**Components:**
- **Vision Encoder**: BiomedCLIP ViT-B/16 (frozen + LoRA adapters on q_proj/v_proj)
- **Text Encoder**: BiomedCLIP PubMedBERT (frozen + LoRA adapters on query/value)
- **Fusion Head**: 2-layer MLP scoring each (image, question, choice) triplet

**Training:**
- LoRA (r=8, alpha=16) on attention projections
- Weighted cross-entropy loss for class imbalance
- Automatic Mixed Precision (AMP) on GPU
- Cosine warmup LR schedule

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train
python src/train.py

# Evaluate + generate explanations
python src/eval.py --checkpoint checkpoints/best.pt --explain --num_explain 10
```

Or run everything:
```bash
bash run.sh
```

### Configuration

All hyperparameters are set in [`src/config.py`](src/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Per GPU batch size |
| `num_epochs` | 15 | Training epochs |
| `lr` | 5e-5 | LoRA parameter learning rate |
| `head_lr` | 1e-3 | Fusion head learning rate |
| `use_lora` | True | Enable LoRA fine-tuning |
| `lora_r` | 8 | LoRA rank |

---

## Project Structure

```
VQA/
├── src/
│   ├── config.py        # Hyperparameters & paths
│   ├── dataset.py       # PMCVQADataset, collate_fn
│   ├── model.py         # BiomedCLIP + LoRA + fusion head
│   ├── train.py         # Training loop with AMP & validation
│   ├── eval.py          # Test evaluation & metrics
│   ├── explain.py       # XAI: attention heatmaps & gradient saliency
│   └── eda.py           # Exploratory Data Analysis
├── requirements.txt
├── run.sh               # Single entry point
└── README.md
```

---

## Explainable AI (XAI)

When running with `--explain`, the model generates a 2×3 grid visualization for each test sample:

| Panel | Content |
|-------|---------|
| **(0,0)** Original image | The medical image |
| **(0,1)** ViT Attention Heatmap | Attention rollout — shows which image regions the model focused on |
| **(0,2)** Gradient Saliency | Gradient-based input sensitivity map |
| **(1,0)** Question | The question text displayed clearly |
| **(1,1)** Choices & Probabilities | Each choice with predicted probability; green = ground truth, red = model prediction |
| **(1,2)** Summary | Ground truth, prediction, and full probability distribution |

Output: `outputs/explanations/{Figure_path}.png`

### Techniques

**ViT Attention Rollout**
- Aggregates attention weights across all transformer layers
- Multiplies attention matrices with residual connections
- Extracts CLS token's attention to image patches

**Gradient Saliency**
- Gradient of the prediction score w.r.t. input pixels
- Highlights image regions most sensitive to the model's decision

---

## Results

*(to be filled)*

| Metric | Value |
|--------|-------|
| Test Accuracy | — |
| Class A Accuracy | — |
| Class B Accuracy | — |
| Class C Accuracy | — |
| Class D Accuracy | — |

---

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA recommended)
- GPU with 8GB+ VRAM recommended
- See [`requirements.txt`](requirements.txt) for full list

---

## References

- [PMC-VQA Dataset](https://github.com/xiaoyi12/PMC-VQA)
- [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
