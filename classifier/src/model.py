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
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2Model
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
                 dropout: float = 0.1, pretrained: bool = True):
        super().__init__()
        # HuggingFace에서 사전학습된 Wav2Vec2 인코더를 불러온다(백본).
        # 학습 시에는 pretrained=True로 사전학습 가중치를 받아 미세조정한다.
        # 추론 시에는(pretrained=False) 골격만 config로 만들고 곧바로 우리
        # safetensors 가중치로 덮어쓰므로, ~360MB짜리 base 백본을 HF에서
        # 다시 받을 필요가 없다(콜드스타트 단축).
        # Both paths build the positional conv under the legacy weight_norm so the
        # pretrained checkpoint loads (training) and our saved checkpoint reloads
        # (inference) with matching state-dict keys. See _legacy_weight_norm.
        # 학습·추론 양쪽 모두 구형 weight_norm으로 pos_conv를 만들어 키를 일치시킨다.
        with _legacy_weight_norm():
            if pretrained:
                self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
            else:
                self.wav2vec2 = Wav2Vec2Model(Wav2Vec2Config.from_pretrained(model_name))
        hidden = self.wav2vec2.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        # 분류를 위한 단일 선형층(헤드). hidden 차원 -> 클래스 개수로 매핑.
        self.classifier = nn.Linear(hidden, num_labels)
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
    def forward(self, input_values, attention_mask=None, labels=None,
                output_frame_logits: bool = False):
        # input_values: 전처리된(정규화/패딩된) 오디오 파형 배치
        # attention_mask: 패딩된 부분을 구분하기 위한 마스크 (1=실제 데이터, 0=패딩)
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state                    # [B, T, H]
        # 인코더의 마지막 은닉 상태(시퀀스 표현)에 드롭아웃 후 분류 헤드를 적용.
        # -> 모든 시간 프레임에 대해 클래스별 로짓을 계산.
        frame_logits = self.classifier(self.dropout(hidden))  # [B, T, C]

        if attention_mask is not None:
            # attention_mask는 원본 파형 길이 기준이므로, 이를 CNN 특징 추출기가
            # 만들어내는 축소된 시간 축(frame 개수)에 맞게 다시 변환해야 한다.
            feat_mask = self.wav2vec2._get_feature_vector_attention_mask(
                frame_logits.shape[1], attention_mask
            ).unsqueeze(-1)                                    # [B, T, 1]
            # 패딩된 프레임을 제외하고, 실제 유효한 프레임들의 로짓만 합산.
            summed = (frame_logits * feat_mask).sum(dim=1)
            # 각 샘플별 유효 프레임 개수(0으로 나누기 방지를 위해 최소 1로 clamp).
            counts = feat_mask.sum(dim=1).clamp(min=1)
            # 패딩을 제외한 "마스킹된 평균"으로 발화 전체 로짓을 얻는다.
            logits = summed / counts                          # [B, C]
        else:
            # 마스크가 없는 경우(전부 유효한 배치 등)에는 단순 평균.
            logits = frame_logits.mean(dim=1)

        loss = None
        if labels is not None:
            # 학습 시에는 발화 레벨 로짓과 정답 레이블로 교차 엔트로피 손실을 계산.
            loss = nn.functional.cross_entropy(logits, labels)

        return AccentOutput(
            loss=loss,
            logits=logits,
            # 프레임 단위 출력은 요청했을 때만 반환(불필요한 메모리 사용 방지).
            frame_logits=frame_logits if output_frame_logits else None,
        )


def build_config(model_name: str = MODEL_NAME) -> Wav2Vec2Config:
    # 레이블 정보가 포함된 Wav2Vec2Config 객체를 생성해서 반환하는 헬퍼.
    # (모델 저장/공유 시 config만 별도로 필요할 때 사용)
    cfg = Wav2Vec2Config.from_pretrained(model_name)
    cfg.num_labels = NUM_LABELS
    cfg.id2label = {int(k): v for k, v in ID2LABEL.items()}
    cfg.label2id = dict(LABEL2ID)
    return cfg
