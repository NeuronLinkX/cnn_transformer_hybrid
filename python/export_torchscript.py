"""학습된 모델을 C++ 추론용 TorchScript로 내보낸다."""

# 타입 힌트의 지연 평가를 사용한다.
from __future__ import annotations

# 명령행 옵션 처리를 위해 argparse를 가져온다.
import argparse
# 파일 경로 처리를 위해 Path를 가져온다.
from pathlib import Path
# 설정 딕셔너리 타입 표기를 위해 Any를 가져온다.
from typing import Any

# 모델 로딩과 tracing을 위해 torch를 가져온다.
import torch
# 설정 파싱을 위해 yaml을 가져온다.
import yaml

# 유연한 export를 위해 베이스라인 모델 생성 함수를 가져온다.
from models.baselines import create_resnet18, create_vit_small
# 주요 경로인 하이브리드 모델 생성 함수를 가져온다.
from models.hybrid import CNNTransformerHybrid


def parse_args() -> argparse.Namespace:
    """TorchScript export용 명령행 인자를 파싱한다."""
    # export용 인자 파서를 생성한다.
    parser = argparse.ArgumentParser(description="Export a trained CIFAR-10 model to TorchScript.")
    # YAML 설정 파일 경로를 입력받는다.
    parser.add_argument("--config", type=str, default="configs/cifar10_hybrid_smoke.yaml")
    # 체크포인트 경로를 입력받는다.
    parser.add_argument("--checkpoint", type=str, default="runs/smoke_test/final_hybrid.pth")
    # 체크포인트 복원을 위한 모델 종류를 입력받는다.
    parser.add_argument(
        "--model",
        type=str,
        default="hybrid",
        choices=("hybrid", "resnet18", "vit_small"),
    )
    # 출력 TorchScript 경로를 입력받는다.
    parser.add_argument("--output", type=str, default="hybrid_cifar10_smoke_ts.pt")
    # LibTorch CPU 추론을 단순하게 유지하기 위해 기본값은 CPU export로 둔다.
    parser.add_argument("--device", type=str, default="cpu")
    # 파싱된 인자를 반환한다.
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    """YAML 설정 파일을 불러온다."""
    # 설정 파일을 UTF-8 인코딩으로 연다.
    with open(path, "r", encoding="utf-8") as fp:
        # YAML을 파이썬 딕셔너리로 파싱한다.
        return yaml.safe_load(fp)


def build_model(model_name: str, cfg: dict[str, Any]):
    """가중치 로딩 전에 올바른 모델 구조를 다시 생성한다."""
    # 중첩된 모델 설정을 한 번만 읽는다.
    mcfg = cfg["model"]

    # export를 위해 하이브리드 모델을 다시 생성한다.
    if model_name == "hybrid":
        return CNNTransformerHybrid(
            backbone_name=mcfg["backbone"],
            num_classes=cfg["num_classes"],
            pretrained=False,
            embed_dim=mcfg["embed_dim"],
            transformer_depth=mcfg["transformer_depth"],
            num_heads=mcfg["num_heads"],
            mlp_ratio=mcfg["mlp_ratio"],
            dropout=mcfg["dropout"],
        )

    # export를 위해 ResNet-18 베이스라인을 다시 생성한다.
    if model_name == "resnet18":
        return create_resnet18(num_classes=cfg["num_classes"], pretrained=False)

    # export를 위해 ViT-small 베이스라인을 다시 생성한다.
    if model_name == "vit_small":
        return create_vit_small(num_classes=cfg["num_classes"], pretrained=False)

    # 지원하지 않는 선택지에는 명확한 에러를 발생시킨다.
    raise ValueError(f"Unsupported model choice: {model_name}")


def main() -> None:
    """학습된 체크포인트를 불러와 TorchScript로 내보낸다."""
    # export용 명령행 인자를 파싱한다.
    args = parse_args()
    # YAML 설정 파일을 불러온다.
    cfg = load_config(args.config)
    # export에 사용할 장치를 결정한다.
    device = torch.device(args.device)
    # 대상 장치 위에 모델 구조를 다시 생성한다.
    model = build_model(args.model, cfg).to(device)
    # 체크포인트 payload 또는 단순 state_dict를 불러온다.
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # 추가 정보가 있는 체크포인트와 순수 state_dict 둘 다 지원한다.
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # 모델 가중치를 복원한다.
    model.load_state_dict(state_dict)
    # 모델을 추론 모드로 전환한다.
    model.eval()

    # 설정된 이미지 크기에 맞는 더미 trace 입력을 만든다.
    example = torch.randn(1, 3, cfg["image_size"], cfg["image_size"], device=device)

    # 최근 PyTorch 문서에서는 TorchScript를 권장하지 않는다.
    # 여기서는 단순한 LibTorch 기반 과제 추론 경로를 위해서만 사용한다.
    # 장기 배포 용도라면 요구사항에 따라 torch.export 또는 ONNX를 검토한다.
    with torch.no_grad():
        # 대표 더미 텐서로 모델을 trace한다.
        traced = torch.jit.trace(model, example)
        # trace된 모듈을 디스크에 저장한다.
        traced.save(args.output)

    # 편의를 위해 최종 출력 경로를 출력한다.
    print(f"saved: {args.output}")


# 스크립트가 직접 실행될 때만 export 경로를 수행한다.
if __name__ == "__main__":
    # TorchScript export 워크플로에 진입한다.
    main()
