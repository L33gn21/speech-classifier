"""Wav2Vec2 backbone + frame-level linear head for accent classification.

Design note (Level 2 readiness):
The linear head is applied to *every frame* -> `frame_logits` [B, T, C]. The
utterance-level `logits` [B, C] are the masked mean of `frame_logits` over
time. Because the head is a single linear layer, this equals "mean-pool the
representations, then apply the head" — so we keep the frame-level output for
free (time-axis accent heatmap in Level 2) while training on utterance labels.
"""
# Wav2Vec2 백본(사전학습 음성 인코더) + 프레임 단위 선형 분류 헤드로 구성된
# 억양 분류 모델 정의.
#
# 설계 노트 (Level 2 대비):
# 분류 헤드(선형층)를 오디오의 "모든 프레임"에 적용하여 frame_logits [B, T, C]를
# 얻는다. 발화(utterance) 전체에 대한 logits [B, C]는 이 frame_logits를
# 시간 축으로 마스킹된 평균(masked mean)을 취해서 얻는다.
# 헤드가 단일 선형층이기 때문에 이는 수학적으로 "표현을 먼저 평균 풀링한 뒤
# 헤드를 적용하는 것"과 동일하다. 따라서 발화 레벨 레이블로 학습하면서도,
# 추가 비용 없이 프레임 단위 출력(Level 2에서 쓰일 시간축 억양 히트맵 재료)을
# 함께 얻을 수 있다.
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
    # transformers 4.44 + torch 2.x 조합에서 wav2vec2 위치정보 conv(pos_conv_embed)
    # 가중치가 사전학습 체크포인트에서 로드되지 않고 랜덤 초기화되는 버그를 회피한다.
    # wav2vec2는 절대 위치 임베딩이 없어 이 conv가 유일한 위치 신호이므로, 랜덤이면
    # 사전학습 표현이 깨져 학습 손실이 ln(클래스수)에 고정된다(학습 불가).
    # 학습(사전학습 로드)과 추론(config만으로 골격 생성) 양쪽에서 동일하게 구형
    # weight_norm(weight_g/weight_v)을 쓰게 해, 저장한 체크포인트와 키 레이아웃을
    # 일치시킨다. 컨텍스트를 벗어나면 즉시 원복한다.
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
    # 모델의 forward() 반환값을 담는 컨테이너.
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None            # [B, C] utterance-level
    # 발화(문장) 전체에 대한 클래스별 로짓. [배치, 클래스수]
    frame_logits: torch.FloatTensor | None = None      # [B, T, C] (opt-in)
    # 프레임(시간 스텝)별 클래스 로짓. 요청 시에만 계산되어 채워짐. [배치, 시간, 클래스수]


class AccentClassifier(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME, num_labels: int = NUM_LABELS,
                 dropout: float = 0.1, pretrained: bool = True,
                 head: str = "mean", layer_weighting: bool = False):
        super().__init__()
        # 사전학습된 음성 인코더(백본)를 불러온다. AutoModel 을 쓰므로 model_name 이
        # facebook/wav2vec2-* 이면 Wav2Vec2Model, microsoft/wavlm-* 이면 WavLMModel 이
        # 자동으로 선택된다. 두 백본은 API(hidden_size, encoder.layers, feature_extractor,
        # _get_feature_vector_attention_mask, output_hidden_states)가 동일해 아래 코드가
        # 그대로 동작한다. 속성 이름은 백본 종류와 무관하게 wav2vec2 로 유지해
        # state_dict 키 접두사(wav2vec2.*)를 안정적으로 둔다.
        # 학습 시에는 pretrained=True로 사전학습 가중치를 받아 미세조정한다.
        # 추론 시에는(pretrained=False) 골격만 config로 만들고 곧바로 우리
        # safetensors 가중치로 덮어쓰므로, base 백본을 HF에서 다시 받을 필요가 없다.
        # Both paths build the positional conv under the legacy weight_norm so the
        # pretrained checkpoint loads (training) and our saved checkpoint reloads
        # (inference) with matching state-dict keys. See _legacy_weight_norm.
        # 학습·추론 양쪽 모두 구형 weight_norm으로 pos_conv를 만들어 키를 일치시킨다.
        with _legacy_weight_norm():
            if pretrained:
                self.wav2vec2 = AutoModel.from_pretrained(model_name)
            else:
                self.wav2vec2 = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        hidden = self.wav2vec2.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        # -- head selection -----------------------------------------------------
        # head="mean": 프레임별 선형 헤드 후 마스킹 평균(=표현 평균 후 헤드와 동일,
        #   단일 선형이라). Level 2 프레임 히트맵을 공짜로 얻는 기존 방식.
        # head="attentive": Attentive Statistics Pooling — 프레임에 어텐션 가중치를
        #   학습해 가중 평균 μ 와 가중 표준편차 σ 를 구하고 [μ;σ] 로 분류한다. 억양은
        #   발음의 "분포/변동"에 정보가 있어 평균만 쓰는 것보다 유리(화자/언어ID 정석).
        self.head_type = head
        # layer_weighting: 마지막 레이어만 쓰지 않고 전 트랜스포머 레이어 은닉상태의
        #   학습가능 가중합을 표현으로 쓴다(SUPERB식). 억양·음소 정보가 중간 레이어에
        #   많으므로 도움이 된다.
        self.layer_weighting = layer_weighting
        if layer_weighting:
            n_states = self.wav2vec2.config.num_hidden_layers + 1  # +1: embedding 출력
            self.layer_weights = nn.Parameter(torch.zeros(n_states))
        if head == "attentive":
            self.attn = nn.Linear(hidden, 1)          # 프레임별 어텐션 스코어
            self.classifier = nn.Linear(hidden * 2, num_labels)  # [μ; σ] -> 클래스
        elif head == "mean":
            # 분류를 위한 단일 선형층(헤드). hidden 차원 -> 클래스 개수로 매핑.
            self.classifier = nn.Linear(hidden, num_labels)
        else:
            raise ValueError(f"unknown head type: {head}")
        self.num_labels = num_labels
        # keep label maps on the module for saving/loading
        # 모델 저장/로드 시 레이블 정보도 함께 보존하기 위해 config에 기록해둔다.
        self.config = self.wav2vec2.config
        self.config.num_labels = num_labels
        self.config.id2label = {int(k): v for k, v in ID2LABEL.items()}
        self.config.label2id = dict(LABEL2ID)
        # conv feature encoder is always frozen (standard for wav2vec2 fine-tuning)
        # 원시 파형을 특징으로 변환하는 CNN 인코더 부분은 항상 동결(freeze)한다.
        # 이는 wav2vec2 파인튜닝에서 일반적으로 쓰이는 관례다.
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
        # 백본의 그래디언트 체크포인팅을 강제로 끈다. wav2vec2-base config에
        # gradient_checkpointing=True 가 들어 있어 자동 활성화되는데, reentrant
        # 모드에서 하위 동결 레이어의 체크포인트 입력이 grad를 요구하지 않으면
        # 그래디언트가 조용히 끊겨("Gradients will be None") 상위 언프리즈 레이어가
        # 학습되지 않는다. 필요할 때만 TrainingArguments 로 use_reentrant=False로
        # 명시 활성화하므로, 기본은 여기서 꺼 둔다.
        self.wav2vec2.config.gradient_checkpointing = False
        try:
            self.wav2vec2.gradient_checkpointing_disable()
        except Exception:
            pass

    # -- freezing helpers -----------------------------------------------------
    # -- 파라미터 동결/해제 관련 헬퍼 함수들 -----------------------------------
    def freeze_backbone(self) -> None:
        # wav2vec2 백본 전체 파라미터를 학습 대상에서 제외(동결)한다.
        # 기본 학습 방식: 헤드(분류층)만 학습.
        for p in self.wav2vec2.parameters():
            p.requires_grad = False

    def unfreeze_top_layers(self, n: int) -> None:
        """Unfreeze the top `n` transformer encoder layers (+ their layer norm)."""
        # 백본을 먼저 전부 동결한 뒤, 상위(마지막) n개의 트랜스포머 인코더
        # 레이어만 다시 학습 가능하도록 해제한다. 헤드만으로는 성능이 부족할 때
        # 점진적으로 미세조정(fine-tuning) 범위를 넓히는 용도.
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
        # 백본 내장 SpecAugment(학습 시 특징 시간/채널 축 마스킹) 세기를 조절한다.
        # 기본으로 켜져 있으나 확률이 낮다 — 값을 올리면 채널 confound 에 대한 사실상
        # 공짜 정규화가 된다. None 이면 백본 기본값 유지.
        cfg = self.wav2vec2.config
        cfg.apply_spec_augment = True
        if time_prob is not None:
            cfg.mask_time_prob = time_prob
        if feature_prob is not None:
            cfg.mask_feature_prob = feature_prob

    # -- gradient checkpointing (delegated to the backbone) -------------------
    # -- 그래디언트 체크포인팅 (백본에게 위임) ---------------------------------
    # HF Trainer calls these on the top-level model when
    # TrainingArguments(gradient_checkpointing=True); forward them to wav2vec2.
    # HuggingFace Trainer는 TrainingArguments(gradient_checkpointing=True)일 때
    # 최상위 모델 객체의 이 메서드들을 호출한다. 우리 모델은 자체 구현이 없으므로
    # 실제로는 내부 wav2vec2 백본에게 그대로 위임(forward)한다.
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self.wav2vec2.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self) -> None:
        self.wav2vec2.gradient_checkpointing_disable()

    # -- forward --------------------------------------------------------------
    # -- 순전파(forward) --------------------------------------------------------
    def _pooled_representation(self, input_values, attention_mask):
        """Run the backbone and return the frame representation + frame mask.

        Returns ``hidden`` [B, T, H] (a learned layer-weighted sum when
        ``layer_weighting``, else the last hidden state) and a boolean
        ``frame_mask`` [B, T] (True on real frames) or None when unmasked.
        """
        # 백본을 돌려 프레임 표현과 프레임 마스크를 만든다. layer_weighting 이면 전
        # 레이어 은닉상태의 학습가능 가중합을, 아니면 마지막 은닉상태를 표현으로 쓴다.
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
            # attention_mask는 원본 파형 길이 기준이므로, CNN 특징추출기가 만드는
            # 축소된 시간 축(frame 개수)에 맞게 다시 변환한다.
            frame_mask = self.wav2vec2._get_feature_vector_attention_mask(
                hidden.shape[1], attention_mask
            ).bool()                                          # [B, T]
        return hidden, frame_mask

    def forward(self, input_values, attention_mask=None, labels=None,
                output_frame_logits: bool = False):
        # input_values: 전처리된(정규화/패딩된) 오디오 파형 배치
        # attention_mask: 패딩된 부분을 구분하기 위한 마스크 (1=실제 데이터, 0=패딩)
        hidden, frame_mask = self._pooled_representation(input_values, attention_mask)
        frame_logits = None

        if self.head_type == "attentive":
            # Attentive Statistics Pooling: 프레임 어텐션 가중치 α 로 가중 평균 μ 와
            # 가중 표준편차 σ 를 구해 [μ; σ] 로 분류한다. 패딩 프레임은 마스킹된
            # softmax 로 가중치 0 이 되어 자연히 제외된다.
            scores = self.attn(hidden).squeeze(-1)            # [B, T]
            if frame_mask is not None:
                scores = scores.masked_fill(~frame_mask, float("-inf"))
            alpha = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]
            mu = (alpha * hidden).sum(dim=1)                  # [B, H]
            var = (alpha * hidden.pow(2)).sum(dim=1) - mu.pow(2)
            sigma = torch.sqrt(var.clamp(min=1e-6))           # [B, H]
            pooled = torch.cat([mu, sigma], dim=-1)           # [B, 2H]
            logits = self.classifier(self.dropout(pooled))    # [B, C]
            # attentive 헤드에서는 발화 로짓이 프레임 로짓의 단순 평균이 아니므로
            # Level 2 프레임 히트맵(단일 선형 투영)이 성립하지 않는다 → None.
        else:  # "mean"
            # 인코더 표현에 드롭아웃 후 프레임별 선형 헤드를 적용해 프레임 로짓을 얻고,
            # 패딩을 제외한 마스킹 평균으로 발화 전체 로짓을 얻는다(단일 선형이라
            # "표현을 평균낸 뒤 헤드 적용"과 동일 → 프레임 히트맵을 공짜로 얻음).
            frame_logits = self.classifier(self.dropout(hidden))  # [B, T, C]
            if frame_mask is not None:
                m = frame_mask.unsqueeze(-1)                  # [B, T, 1]
                summed = (frame_logits * m).sum(dim=1)
                counts = m.sum(dim=1).clamp(min=1)
                logits = summed / counts                      # [B, C]
            else:
                logits = frame_logits.mean(dim=1)

        loss = None
        if labels is not None:
            # 학습 시에는 발화 레벨 로짓과 정답 레이블로 교차 엔트로피 손실을 계산.
            loss = nn.functional.cross_entropy(logits, labels)

        return AccentOutput(
            loss=loss,
            logits=logits,
            # 프레임 단위 출력은 요청 시 + mean 헤드일 때만 채워진다(불필요한 메모리 절약).
            frame_logits=frame_logits if output_frame_logits else None,
        )


def build_config(model_name: str = MODEL_NAME):
    # 레이블 정보가 포함된 백본 config 객체를 생성해서 반환하는 헬퍼.
    # (모델 저장/공유 시 config만 별도로 필요할 때 사용)
    cfg = AutoConfig.from_pretrained(model_name)
    cfg.num_labels = NUM_LABELS
    cfg.id2label = {int(k): v for k, v in ID2LABEL.items()}
    cfg.label2id = dict(LABEL2ID)
    return cfg


# --- architecture persistence / reload ---------------------------------------
# 학습 때 쓴 아키텍처(백본·헤드·레이어가중·dropout)를 저장/복원하는 헬퍼.
# infer.py / evaluate.py / model_tester 는 저장된 가중치와 정확히 같은 구조로
# 모델을 재구성해야 load_state_dict 가 맞는다. 학습 시 write_model_config() 로
# model_config.json 을 남기고, 로드 시 load_from_dir() 로 같은 구조를 만든다.
MODEL_CONFIG_FILE = "model_config.json"


def write_model_config(model_dir, *, backbone: str, num_labels: int,
                       dropout: float, head: str, layer_weighting: bool) -> None:
    """Persist the arch hyperparameters needed to rebuild this model for inference."""
    cfg = {
        "backbone": backbone,
        "num_labels": num_labels,
        "dropout": dropout,
        "head": head,
        "layer_weighting": layer_weighting,
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
    # 저장된 체크포인트와 동일한 구조로 모델을 재구성한다. model_config.json 이
    # 있으면 그 값을, 없으면(구버전) 레거시 기본값(wav2vec2-base/mean/가중없음)을 쓴다.
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
    )
