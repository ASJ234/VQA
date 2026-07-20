import math
import torch
import torch.nn as nn
from transformers import AutoModel


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


def _get_pooled(x):
    if isinstance(x, torch.Tensor):
        return x[:, 0] if x.dim() == 3 else x
    if hasattr(x, 'pooler_output') and x.pooler_output is not None:
        return x.pooler_output
    if hasattr(x, 'last_hidden_state'):
        return x.last_hidden_state[:, 0]
    raise TypeError(f"Unexpected encoder output type: {type(x)}")


class PMCVQAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone = AutoModel.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )

        vision_dim = 768
        text_dim = 768

        if config.use_lora:
            _replace_linear_with_lora(
                self.backbone.vision_encoder, {'q_proj', 'v_proj'}, config)
            _replace_linear_with_lora(
                self.backbone.text_encoder, {'query', 'value'}, config)
        else:
            for p in self.backbone.vision_encoder.parameters():
                p.requires_grad = False
            for p in self.backbone.text_encoder.parameters():
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

        v_out = self.backbone.vision_encoder(images)
        img_feat = _get_pooled(v_out).float()

        q_out = self.backbone.text_encoder(
            question_input_ids, attention_mask=question_attention_mask)
        q_feat = _get_pooled(q_out).float()

        c_ids = choices_input_ids.view(B * 4, -1)
        c_mask = choices_attention_mask.view(B * 4, -1)
        c_out = self.backbone.text_encoder(c_ids, attention_mask=c_mask)
        c_feat = _get_pooled(c_out).float().view(B, 4, -1)

        img_feat = img_feat.unsqueeze(1).expand(-1, 4, -1)
        q_feat = q_feat.unsqueeze(1).expand(-1, 4, -1)

        fusion_in = torch.cat([img_feat, q_feat, c_feat], dim=-1)
        scores = self.fusion_head(fusion_in).squeeze(-1)
        return scores


def get_fusion_head_params(model):
    return [p for n, p in model.named_parameters() if 'fusion_head' in n]


def get_lora_params(model):
    return [p for n, p in model.named_parameters() if 'lora_' in n]


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
