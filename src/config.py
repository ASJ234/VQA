from dataclasses import dataclass


@dataclass
class Config:
    # Paths
    data_root: str = "pmc-vqa_data"
    image_dir: str = "pmc-vqa_data/images_2/figures"
    train_csv: str = "pmc-vqa_data/train_2.csv"
    test_csv: str = "pmc-vqa_data/test_2.csv"
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    explain_dir: str = "outputs/explanations"

    # Model
    model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1

    # Fusion head
    fusion_hidden: int = 512
    fusion_dropout: float = 0.3

    # Training
    batch_size: int = 32
    num_epochs: int = 15
    lr: float = 5e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Data
    image_size: int = 224
    max_text_length: int = 128
    num_workers: int = 4
    val_split: float = 0.05
    pin_memory: bool = True
    max_train_samples: int = 0  # 0 = use all, e.g. 10000 for fast iteration

    # Precision
    use_amp: bool = True

    # Evaluation
    eval_batch_size: int = 64
    num_explain_samples: int = 10
