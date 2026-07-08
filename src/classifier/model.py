"""Wav2Vec2 backbone + frame-level linear head for accent classification.

Design note (Level 2 readiness, per CLAUDE.md):
The linear head is applied to *every frame* -> `frame_logits` [B, T, C]. The
utterance-level `logits` [B, C] are the masked mean of `frame_logits` over
time. Because the head is a single linear layer, this equals "mean-pool the
representations, then apply the head" — so we keep the frame-level output for
free (time-axis accent heatmap in Level 2) while training on utterance labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2Model
from transformers.modeling_outputs import ModelOutput

from config import ID2LABEL, LABEL2ID, MODEL_NAME, NUM_LABELS


@dataclass
class AccentOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None            # [B, C] utterance-level
    frame_logits: torch.FloatTensor | None = None      # [B, T, C] (opt-in)


class AccentClassifier(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME, num_labels: int = NUM_LABELS,
                 dropout: float = 0.1):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        hidden = self.wav2vec2.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.num_labels = num_labels
        # keep label maps on the module for saving/loading
        self.config = self.wav2vec2.config
        self.config.num_labels = num_labels
        self.config.id2label = {int(k): v for k, v in ID2LABEL.items()}
        self.config.label2id = dict(LABEL2ID)
        # conv feature encoder is always frozen (standard for wav2vec2 fine-tuning)
        self.wav2vec2.feature_extractor._freeze_parameters()

    # -- freezing helpers -----------------------------------------------------
    def freeze_backbone(self) -> None:
        for p in self.wav2vec2.parameters():
            p.requires_grad = False

    def unfreeze_top_layers(self, n: int) -> None:
        """Unfreeze the top `n` transformer encoder layers (+ their layer norm)."""
        self.freeze_backbone()
        layers = self.wav2vec2.encoder.layers
        for layer in layers[len(layers) - n:]:
            for p in layer.parameters():
                p.requires_grad = True

    # -- gradient checkpointing (delegated to the backbone) -------------------
    # HF Trainer calls these on the top-level model when
    # TrainingArguments(gradient_checkpointing=True); forward them to wav2vec2.
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self.wav2vec2.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self) -> None:
        self.wav2vec2.gradient_checkpointing_disable()

    # -- forward --------------------------------------------------------------
    def forward(self, input_values, attention_mask=None, labels=None,
                output_frame_logits: bool = False):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state                    # [B, T, H]
        frame_logits = self.classifier(self.dropout(hidden))  # [B, T, C]

        if attention_mask is not None:
            feat_mask = self.wav2vec2._get_feature_vector_attention_mask(
                frame_logits.shape[1], attention_mask
            ).unsqueeze(-1)                                    # [B, T, 1]
            summed = (frame_logits * feat_mask).sum(dim=1)
            counts = feat_mask.sum(dim=1).clamp(min=1)
            logits = summed / counts                          # [B, C]
        else:
            logits = frame_logits.mean(dim=1)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return AccentOutput(
            loss=loss,
            logits=logits,
            frame_logits=frame_logits if output_frame_logits else None,
        )


def build_config(model_name: str = MODEL_NAME) -> Wav2Vec2Config:
    cfg = Wav2Vec2Config.from_pretrained(model_name)
    cfg.num_labels = NUM_LABELS
    cfg.id2label = {int(k): v for k, v in ID2LABEL.items()}
    cfg.label2id = dict(LABEL2ID)
    return cfg
