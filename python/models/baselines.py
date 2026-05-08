"""비교 실험용 베이스라인 모델 생성 함수 모음."""

# 타입 힌트의 지연 평가를 사용한다.
from __future__ import annotations

# 사전학습 베이스라인 모델 생성을 위해 timm을 가져온다.
import timm


def create_resnet18(num_classes: int = 10, pretrained: bool = True):
    """CIFAR-10 미세조정용 ResNet-18 분류기 베이스라인을 반환한다."""
    # timm 생성 과정에서 명확한 오류를 주도록 감싼다.
    try:
        # 일반 ResNet-18 분류기 베이스라인을 생성한다.
        return timm.create_model(
            "resnet18",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    except Exception as exc:  # pragma: no cover - 방어 경로
        # timm 생성이 실패하면 사용자 친화적인 메시지를 발생시킨다.
        raise RuntimeError(
            "Failed to create the ResNet-18 baseline with timm. "
            "Check that timm is installed and the model entry is available."
        ) from exc


def create_vit_small(num_classes: int = 10, pretrained: bool = True):
    """사용 가능한 경우 ViT-Small 베이스라인을 반환한다."""
    # 우선순위 순서대로 자주 쓰이는 ViT-small 모델 이름을 시도한다.
    candidate_names = ("vit_small_patch16_224", "deit_small_patch16_224")
    # 디버깅을 돕기 위해 상세한 생성 오류를 수집한다.
    errors: list[str] = []

    # 알려진 timm ViT-small 식별자를 순회한다.
    for model_name in candidate_names:
        # 현재 후보 모델을 생성해 본다.
        try:
            # 첫 번째로 성공한 후보를 즉시 반환한다.
            return timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
            )
        except Exception as exc:  # pragma: no cover - 방어 경로
            # 최종 예외에 포함할 수 있도록 정확한 실패 내용을 저장한다.
            errors.append(f"{model_name}: {exc}")

    # 모든 후보가 실패하면 하나의 설명적인 오류를 발생시킨다.
    raise RuntimeError(
        "Failed to create a ViT-small baseline. Tried the following timm models: "
        f"{', '.join(candidate_names)}. Details: {' | '.join(errors)}"
    )
