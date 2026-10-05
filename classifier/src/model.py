"""Wav2Vec2 backbone + frame-level linear head for accent classification.

Design note (Level 2 readiness):
The linear head is applied to *every frame* -> `frame_logits` [B, T, C]. The
utterance-level `logits` [B, C] are the masked mean of `frame_logits` over
time. Because the head is a single linear layer, this equals "mean-pool the
representations, then apply the head" — so we keep the frame-level output for
free (time-axis accent heatmap in Level 2) while training on utterance labels.
"""
from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import ModelOutput

from config import ID2LABEL, LABEL2ID, MODEL_NAME, NUM_LABELS


@contextlib.contextmanager
def _legacy_weight_norm():
    """Force wav2vec2's positional conv to use the *legacy* torch weight_norm.

    Bug this works around: transformers 4.44 + torch>=2.1 build the wav2vec2
    positional conv with the *new* ``nn.utils.parametrizations.weight_norm``
    (state-dict keys ``pos_conv_embed.conv.parametrizations.weight.original0/1``),
    but the ``facebook/wav2vec2-base`` checkpoint stores the *legacy* keys
    ``pos_conv_embed.conv.weight_g / weight_v``. The keys don't match, so
    ``from_pretrained`` leaves pos_conv **randomly initialized** (it prints
    "Some weights ... were newly initialized: ... pos_conv_embed..."). wav2vec2
    has no absolute position embedding — this conv is its *only* positional
    signal — so a random pos_conv silently destroys the pretrained
    representation and the classifier cannot learn (training loss stays pinned
    at ln(num_classes)).

    Hiding ``nn.utils.parametrizations.weight_norm`` makes transformers fall back
    to the legacy ``nn.utils.weight_norm`` (weight_g/weight_v). Used for BOTH the
    pretrained load (so the checkpoint keys line up) AND the config-only build
    used at inference (so the module's key layout matches what we saved during
    training). Restored on exit, so nothing else in the process is affected.
    """
    import torch.nn.utils.parametrizations as _param

    saved = getattr(_param, "weight_norm", None)
    try:
        if saved is not None:
            del _param.weight_norm
        yield
    finally:
        if saved is not None:
            _param.weight_norm = saved


@dataclass
class AccentOutput(ModelOutput):
    # Container for the values returned by the model's forward().
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None            # [B, C] utterance-level
    # Per-class logits for the whole utterance. [batch, num_classes]
    fake_logits: torch.FloatTensor | None = None       # [B, 2] real/fake (opt-in)
    # Logits of the binary real/fake head (multitask). Filled only when the fake head
    # is enabled. [batch, 2]
    frame_logits: torch.FloatTensor | None = None      # [B, T, C] (opt-in)
    # Per-frame (time-step) class logits. Computed and filled only on request.
    # [batch, time, num_classes]


class AccentClassifier(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME, num_labels: int = NUM_LABELS,
                 dropout: float = 0.1, pretrained: bool = True,
                 head: str = "mean", layer_weighting: bool = False,
                 fake_head: bool = False, num_fake_labels: int = 2):
        super().__init__()
        # Load the pretrained speech encoder (backbone). Because AutoModel is used,
        # Wav2Vec2Model is selected automatically for facebook/wav2vec2-* names and
        # WavLMModel for microsoft/wavlm-*. Both backbones share the same API (hidden_size,
        # encoder.layers, feature_extractor, _get_feature_vector_attention_mask,
        # output_hidden_states), so the code below works for either. The attribute is named
        # wav2vec2 regardless of the backbone type so the state_dict key prefix
        # (wav2vec2.*) stays stable.
        # In training, pretrained=True loads the pretrained weights for fine-tuning.
        # At inference (pretrained=False) only the skeleton is built from the config and is
        # immediately overwritten with our safetensors weights, so the base backbone never
        # has to be downloaded from HF again.
        # Both paths build the positional conv under the legacy weight_norm so the
        # pretrained checkpoint loads (training) and our saved checkpoint reloads
        # (inference) with matching state-dict keys. See _legacy_weight_norm.
        with _legacy_weight_norm():
            if pretrained:
                self.wav2vec2 = AutoModel.from_pretrained(model_name)
            else:
                self.wav2vec2 = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        hidden = self.wav2vec2.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        # -- head selection -----------------------------------------------------
        # head="mean": per-frame linear head followed by a masked mean (identical to pooling
        #   the representation first and then applying the head, since it is a single linear
        #   layer). The original design; yields the Level 2 frame heatmap for free.
        # head="attentive": Attentive Statistics Pooling — learns attention weights over the
        #   frames, computes the weighted mean μ and weighted standard deviation σ, and
        #   classifies on [μ;σ]. Accent information lies in the distribution/variation of
        #   pronunciation, so this beats the mean alone (the standard in speaker/language ID).
        self.head_type = head
        # layer_weighting: instead of the last layer only, use a learnable weighted sum of
        #   the hidden states of all transformer layers as the representation (SUPERB-style).
        #   Helps because much of the accent/phoneme information lives in the middle layers.
        self.layer_weighting = layer_weighting
        if layer_weighting:
            n_states = self.wav2vec2.config.num_hidden_layers + 1  # +1: embedding output
            self.layer_weights = nn.Parameter(torch.zeros(n_states))
        # Dimension of the pooled utterance representation: hidden for the mean head,
        # 2*hidden ([μ;σ]) for the attentive head.
        pooled_dim = hidden * 2 if head == "attentive" else hidden
        if head == "attentive":
            self.attn = nn.Linear(hidden, 1)          # per-frame attention score
            self.classifier = nn.Linear(pooled_dim, num_labels)  # [μ; σ] -> classes
        elif head == "mean":
            # Single linear layer (head) for classification: hidden dimension -> number of classes.
            self.classifier = nn.Linear(pooled_dim, num_labels)
        else:
            raise ValueError(f"unknown head type: {head}")
        self.num_labels = num_labels

        # -- fake (real/spoof) head (multitask, opt-in) -------------------------
        # Binary real/fake classification on the same pooled utterance representation of the
        # shared backbone. The backbone is shared with the country head; only the heads are
        # separate. With fake_head=False it is not created, so the state_dict is identical to
        # existing country-only checkpoints.
        self.fake_head_enabled = bool(fake_head)
        self.num_fake_labels = num_fake_labels
        if self.fake_head_enabled:
            self.fake_classifier = nn.Linear(pooled_dim, num_fake_labels)
        # keep label maps on the module for saving/loading
        self.config = self.wav2vec2.config
        self.config.num_labels = num_labels
        self.config.id2label = {int(k): v for k, v in ID2LABEL.items()}
        self.config.label2id = dict(LABEL2ID)
        # conv feature encoder is always frozen (standard for wav2vec2 fine-tuning)
        self.wav2vec2.feature_extractor._freeze_parameters()

        # Force gradient checkpointing OFF on the backbone. The wav2vec2-base
        # config ships gradient_checkpointing=True, which gets auto-enabled; in
        # *reentrant* mode (the default; our use_reentrant=False is only wired to
        # the TrainingArguments path, which the default recipe doesn't trigger) a
        # checkpointed segment whose inputs don't require grad silently drops
        # gradients ("None of the inputs have requires_grad=True. Gradients will
        # be None"). With the lower layers frozen that severs gradient flow to the
        # unfrozen top layers, so the model can't fit even a tiny set. We enable
        # checkpointing explicitly (use_reentrant=False) via TrainingArguments
        # only when asked, so keep it off here by default.
        self.wav2vec2.config.gradient_checkpointing = False
        try:
            self.wav2vec2.gradient_checkpointing_disable()
        except Exception:
            pass

    # -- freezing helpers -----------------------------------------------------
    def freeze_backbone(self) -> None:
        # Exclude (freeze) all parameters of the wav2vec2 backbone from training.
        # Default mode: train only the head (classification layer).
        for p in self.wav2vec2.parameters():
            p.requires_grad = False

    def unfreeze_top_layers(self, n: int) -> None:
        """Unfreeze the top `n` transformer encoder layers (+ their layer norm)."""
        # Freeze the whole backbone first, then make only the top (last) n transformer
        # encoder layers trainable again. Used to widen the fine-tuning range step by step
        # when the head alone does not perform well enough.
        self.freeze_backbone()
        layers = self.wav2vec2.encoder.layers
        for layer in layers[len(layers) - n:]:
            for p in layer.parameters():
                p.requires_grad = True

    def set_spec_augment(self, time_prob: float | None = None,
                         feature_prob: float | None = None) -> None:
        """Tune the backbone's native SpecAugment masking (train-time only).

        wav2vec2/wavlm mask spans of the feature-encoder output during training
        (``config.apply_spec_augment``). It is on by default at a low rate; raise
        the mask probabilities for stronger, essentially-free regularization
        against the channel confound. ``None`` leaves the backbone default.
        """
        cfg = self.wav2vec2.config
        cfg.apply_spec_augment = True
        if time_prob is not None:
            cfg.mask_time_prob = time_prob
        if feature_prob is not None:
            cfg.mask_feature_prob = feature_prob

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
    def _pooled_representation(self, input_values, attention_mask):
        """Run the backbone and return the frame representation + frame mask.

        Returns ``hidden`` [B, T, H] (a learned layer-weighted sum when
        ``layer_weighting``, else the last hidden state) and a boolean
        ``frame_mask`` [B, T] (True on real frames) or None when unmasked.
        """
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask,
                                output_hidden_states=self.layer_weighting)
        if self.layer_weighting:
            hs = torch.stack(outputs.hidden_states, dim=0)    # [L+1, B, T, H]
            w = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
            hidden = (hs * w).sum(dim=0)                       # [B, T, H]
        else:
            hidden = outputs.last_hidden_state                # [B, T, H]

        frame_mask = None
        if attention_mask is not None:
            # attention_mask is defined on the raw waveform length, so convert it to the reduced
            # time axis (number of frames) produced by the CNN feature extractor.
            frame_mask = self.wav2vec2._get_feature_vector_attention_mask(
                hidden.shape[1], attention_mask
            ).bool()                                          # [B, T]
        return hidden, frame_mask

    def forward(self, input_values, attention_mask=None, labels=None,
                output_frame_logits: bool = False):
        # input_values: batch of preprocessed (normalized/padded) audio waveforms
        # attention_mask: mask that marks the padded positions (1 = real data, 0 = padding)
        hidden, frame_mask = self._pooled_representation(input_values, attention_mask)
        frame_logits = None

        if self.head_type == "attentive":
            # Attentive Statistics Pooling: with frame attention weights α, compute the
            # weighted mean μ and weighted standard deviation σ and classify on [μ; σ]. Padded
            # frames get weight 0 through the masked softmax and are excluded naturally.
            scores = self.attn(hidden).squeeze(-1)            # [B, T]
            if frame_mask is not None:
                scores = scores.masked_fill(~frame_mask, float("-inf"))
            alpha = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]
            mu = (alpha * hidden).sum(dim=1)                  # [B, H]
            var = (alpha * hidden.pow(2)).sum(dim=1) - mu.pow(2)
            sigma = torch.sqrt(var.clamp(min=1e-6))           # [B, H]
            pooled = torch.cat([mu, sigma], dim=-1)           # [B, 2H]
            logits = self.classifier(self.dropout(pooled))    # [B, C]
            # With the attentive head the utterance logits are not a plain mean of the frame
            # logits, so the Level 2 frame heatmap (a single linear projection) does not hold → None.
        else:  # "mean"
            # Apply dropout to the encoder representation and then the per-frame linear head to
            # get frame logits; the utterance logits are their masked mean excluding padding
            # (with a single linear layer this equals "average the representation, then apply
            # the head" → the frame heatmap comes for free).
            frame_logits = self.classifier(self.dropout(hidden))  # [B, T, C]
            if frame_mask is not None:
                m = frame_mask.unsqueeze(-1)                  # [B, T, 1]
                summed = (frame_logits * m).sum(dim=1)
                counts = m.sum(dim=1).clamp(min=1)
                logits = summed / counts                      # [B, C]
                # pooled for the fake head = masked-mean representation (padding excluded). [B, H]
                pooled = (hidden * m).sum(dim=1) / counts
            else:
                logits = frame_logits.mean(dim=1)
                pooled = hidden.mean(dim=1)

        # real/fake head: binary logits from the shared pooled utterance representation above.
        fake_logits = None
        if self.fake_head_enabled:
            fake_logits = self.fake_classifier(self.dropout(pooled))  # [B, 2]

        loss = None
        if labels is not None:
            # Single-task (country) compatibility path: CE loss from the utterance logits and
            # country labels. The combined multitask (country+fake) loss is handled by
            # MultiTaskTrainer on the training side; in that case this internal loss is not used
            # (labels are not passed).
            loss = nn.functional.cross_entropy(logits, labels)

        return AccentOutput(
            loss=loss,
            logits=logits,
            fake_logits=fake_logits,
            # Frame-level output is filled only on request and only for the mean head (saves memory).
            frame_logits=frame_logits if output_frame_logits else None,
        )


def build_config(model_name: str = MODEL_NAME):
    # Helper that creates and returns a backbone config object including the label information.
    # (used when only the config is needed, e.g. when saving/sharing the model)
    cfg = AutoConfig.from_pretrained(model_name)
    cfg.num_labels = NUM_LABELS
    cfg.id2label = {int(k): v for k, v in ID2LABEL.items()}
    cfg.label2id = dict(LABEL2ID)
    return cfg


# --- architecture persistence / reload ---------------------------------------
# Helpers that save/restore the architecture used in training (backbone, head, layer
# weighting, dropout). infer.py / evaluate.py / model_tester must rebuild the model with
# exactly the same structure as the saved weights for load_state_dict to match. Training
# writes model_config.json via write_model_config(); loading rebuilds it via load_from_dir().
MODEL_CONFIG_FILE = "model_config.json"


def write_model_config(model_dir, *, backbone: str, num_labels: int,
                       dropout: float, head: str, layer_weighting: bool,
                       fake_head: bool = False, num_fake_labels: int = 2) -> None:
    """Persist the arch hyperparameters needed to rebuild this model for inference."""
    cfg = {
        "backbone": backbone,
        "num_labels": num_labels,
        "dropout": dropout,
        "head": head,
        "layer_weighting": layer_weighting,
        # Presence/size of the fake head (multitask). Old checkpoints lack this key and fall
        # back to False on load.
        "fake_head": bool(fake_head),
        "num_fake_labels": num_fake_labels,
    }
    with open(Path(model_dir) / MODEL_CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def load_from_dir(model_dir, num_labels: int | None = None) -> "AccentClassifier":
    """Rebuild an AccentClassifier matching a saved checkpoint's architecture.

    Reads ``model_config.json`` if present (backbone / head / layer_weighting);
    falls back to the legacy default (wav2vec2-base, mean head, no layer
    weighting) for models saved before that file existed. Builds the skeleton
    with ``pretrained=False`` (weights are loaded by the caller). The backbone
    weights come from the caller's ``load_state_dict``, so we never re-download.
    """
    p = Path(model_dir) / MODEL_CONFIG_FILE
    if p.exists():
        cfg = json.loads(p.read_text())
    else:
        cfg = {}
    backbone = cfg.get("backbone", MODEL_NAME)
    n = num_labels if num_labels is not None else cfg.get("num_labels", NUM_LABELS)
    return AccentClassifier(
        backbone,
        num_labels=n,
        dropout=cfg.get("dropout", 0.1),
        pretrained=False,
        head=cfg.get("head", "mean"),
        layer_weighting=cfg.get("layer_weighting", False),
        # Old country-only checkpoints have no fake_head key → falls back to False (same structure).
        fake_head=cfg.get("fake_head", False),
        num_fake_labels=cfg.get("num_fake_labels", 2),
    )
