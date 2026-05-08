"""하이브리드 CNN-Transformer 모델 정의."""

# 더 깔끔한 타입 힌트를 위해 지연 평가를 사용한다.
from __future__ import annotations

# PyTorch 최상위 패키지를 가져온다.
import torch
# 신경망 모듈 네임스페이스를 가져온다.
import torch.nn as nn
# 사전학습 비전 백본을 만들기 위해 timm을 가져온다.
import timm


class CNNTransformerHybrid(nn.Module):
    """
    ResNet-18 CNN 백본과 Transformer Encoder를 결합한 분류기.

    입력:  B x 3 x 224 x 224
    CNN:   B x 512 x 7 x 7
    Token: B x 49 x 512
    CLS+:  B x 50 x 512
    출력:  B x num_classes
    """

    def __init__(
        self,
        # timm 모델 이름으로 CNN 백본을 선택한다.
        backbone_name: str = "resnet18",
        # 최종 분류기의 출력 차원을 설정한다.
        num_classes: int = 10,
        # 가능하면 ImageNet 사전학습 가중치를 사용한다.
        pretrained: bool = True,
        # Transformer 임베딩 차원을 설정한다.
        embed_dim: int = 512,
        # Transformer encoder 레이어 수를 설정한다.
        transformer_depth: int = 2,
        # self-attention head 수를 설정한다.
        num_heads: int = 8,
        # Transformer 블록 내부의 feed-forward 확장 비율을 설정한다.
        mlp_ratio: float = 4.0,
        # Transformer 경로 주변에 사용할 dropout 비율을 설정한다.
        dropout: float = 0.1,
    ) -> None:
        # 기본 nn.Module 상태를 초기화한다.
        super().__init__()

        # 224x224 입력이 7x7 특징으로 바뀔 때의 공간 토큰 수를 기록한다.
        self.num_spatial_tokens = 49
        # 학습 중 백본이 의도적으로 고정되었는지 추적한다.
        self._backbone_frozen = False

        # 분류기 head를 제거하고 공간 feature map만 유지한다.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        # timm 메타데이터에서 백본 출력 채널 수를 읽는다.
        in_ch = getattr(self.backbone, "num_features", None)

        # 선택된 백본이 feature 채널 수를 제공하지 않으면 초기에 실패시킨다.
        if in_ch is None:
            raise ValueError(
                f"Backbone '{backbone_name}' does not expose 'num_features'. "
                "Please use a timm CNN backbone that returns spatial feature maps."
            )

        # CNN 채널 특징을 Transformer 임베딩 공간으로 사영한다.
        self.proj = nn.Linear(in_ch, embed_dim)
        # 시퀀스 수준 분류를 위한 단일 CLS 토큰을 학습한다.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 1개 CLS 토큰과 49개 공간 토큰에 대한 위치 임베딩을 학습한다.
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_spatial_tokens, embed_dim))
        # 토큰과 위치 정보를 합친 뒤 dropout을 적용한다.
        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder 블록 정의를 구성한다.
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # norm_first 경고를 피하기 위해 nested tensor 최적화 없이 encoder를 구성한다.
        try:
            # 가능하면 최신 PyTorch 인자를 사용한다.
            self.encoder = nn.TransformerEncoder(
                enc_layer,
                num_layers=transformer_depth,
                enable_nested_tensor=False,
            )
        except TypeError:
            # 호환성을 위해 이전 생성자 시그니처로 되돌린다.
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=transformer_depth)
        # 분류 전에 CLS 임베딩을 정규화한다.
        self.norm = nn.LayerNorm(embed_dim)
        # CLS 임베딩을 클래스 로짓으로 변환한다.
        self.head = nn.Linear(embed_dim, num_classes)

        # 학습 가능한 토큰 파라미터를 초기화한다.
        self._init_tokens()

    def _init_tokens(self) -> None:
        # CLS 토큰을 작은 truncated normal 분포로 초기화한다.
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # 위치 임베딩을 작은 truncated normal 분포로 초기화한다.
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def freeze_backbone(self, frozen: bool = True) -> None:
        """CNN 백본 파라미터를 고정하거나 해제한다."""
        # train()에서 batchnorm 계층을 안정적으로 유지할 수 있도록 고정 상태를 저장한다.
        self._backbone_frozen = frozen
        # 백본의 모든 파라미터에 대해 gradient 계산 여부를 전환한다.
        for p in self.backbone.parameters():
            p.requires_grad = not frozen

    def train(self, mode: bool = True) -> "CNNTransformerHybrid":
        """백본이 고정된 단계에서는 eval 모드를 유지한다."""
        # 먼저 부모 클래스가 전체 학습 모드를 전환하도록 둔다.
        super().train(mode)
        # 1단계가 활성화되어 있으면 백본의 running statistics를 다시 고정한다.
        if mode and self._backbone_frozen:
            self.backbone.eval()
        # nn.Module.train과 같은 동작을 위해 self를 반환한다.
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 설계된 방식대로 백본 순전파를 수행한다.
        feat = self.backbone(x)

        # 일부 timm 백본은 공간 feature map을 얻기 위해 forward_features가 필요할 수 있다.
        if feat.ndim != 4 and hasattr(self.backbone, "forward_features"):
            feat = self.backbone.forward_features(x)

        # 입력받은 특징이 B x C x H x W 형태인지 검증한다.
        if feat.ndim != 4:
            raise ValueError(
                "Expected backbone feature map with shape B x C x H x W, "
                f"but got {tuple(feat.shape)} instead."
            )

        # feat: B x 512 x 7 x 7 형태이며, 7x7 공간 셀이 Transformer 토큰이 된다.
        # tokens: B x 49 x 512
        # 앞에 CLS 토큰을 붙여 최종 시퀀스 길이는 50이 된다.
        # 참고: 이미지 크기나 백본이 달라지면 토큰 수가 바뀔 수 있다.
        tokens = feat.flatten(2).transpose(1, 2)

        # TorchScript tracing 중에는 파이썬 측 shape 검사 경고를 건너뛴다.
        if (not torch.jit.is_tracing()) and tokens.size(1) != self.num_spatial_tokens:
            raise ValueError(
                f"Expected {self.num_spatial_tokens} spatial tokens, but got {tokens.size(1)}. "
                "If you changed the backbone or image size, update the positional embedding length."
            )

        # CNN 토큰을 Transformer 임베딩 차원으로 사영한다.
        tokens = self.proj(tokens)
        # 현재 배치 크기에 맞게 CLS 토큰을 확장한다.
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        # 앞에 CLS 토큰을 붙여 시퀀스를 B x 50 x D로 만든다.
        x = torch.cat([cls, tokens], dim=1)
        # 토큰 순서 정보를 보존하기 위해 위치 임베딩을 더한다.
        x = x + self.pos_embed[:, : x.size(1), :]
        # 위치 인코딩 뒤에 dropout을 적용한다.
        x = self.pos_drop(x)
        # 전체 토큰 시퀀스를 Transformer encoder에 통과시킨다.
        x = self.encoder(x)
        # CLS 토큰 출력을 추출하고 정규화한다.
        cls_out = self.norm(x[:, 0])
        # 정규화된 CLS 임베딩을 클래스 로짓으로 변환한다.
        return self.head(cls_out)
