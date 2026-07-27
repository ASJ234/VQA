import math
import torch
import torch.nn as nn
import open_clip


class LoRAAdapter(nn.Module):
    def __init__(self, linear, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.linear = linear
        self.scaling = alpha / r
        in_feat = linear.in_features
        out_feat = linear.out_features

        self.lora_A = nn.Parameter(torch.empty(r, in_feat))
        self.lora_B = nn.Parameter(torch.empty(out_feat, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling


def _replace_linear_with_lora(module, target_names, config):
    for name, child in module.named_children():
        if name in target_names and isinstance(child, nn.Linear):
            setattr(module, name,
                    LoRAAdapter(child, config.lora_r, config.lora_alpha, config.lora_dropout))
        else:
            _replace_linear_with_lora(child, target_names, config)


class PMCVQAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone, _, _ = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        vision_dim = 512
        text_dim = 512

        for p in self.backbone.parameters():
            p.requires_grad = False

        fusion_dim = vision_dim + text_dim * 2
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, config.fusion_hidden),
            nn.GELU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.fusion_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, images, question_input_ids, question_attention_mask,
                choices_input_ids, choices_attention_mask):
        B = images.size(0)

        img_feat = self.backbone.encode_image(images).float()

        q_feat = self.backbone.encode_text(question_input_ids).float()

        c_ids = choices_input_ids.view(B * 4, -1)
        c_feat = self.backbone.encode_text(c_ids).float().view(B, 4, -1)

        img_feat = img_feat.unsqueeze(1).expand(-1, 4, -1)
        q_feat = q_feat.unsqueeze(1).expand(-1, 4, -1)

        fusion_in = torch.cat([img_feat, q_feat, c_feat], dim=-1)
        scores = self.fusion_head(fusion_in).squeeze(-1)
        return scores


def get_fusion_head_params(model):
    return [p for n, p in model.named_parameters() if 'fusion_head' in n]


def get_lora_params(model):
    return []


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
