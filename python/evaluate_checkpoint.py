"""체크포인트를 CIFAR-10 전체 테스트셋에서 평가한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from train import build_model, get_device, load_config
from utils.data import build_loaders
from utils.metrics import evaluate


def parse_args() -> argparse.Namespace:
    """명령행 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="Evaluate a CIFAR-10 checkpoint.")
    parser.add_argument("--config", type=str, default="configs/cifar10_hybrid_full.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="hybrid",
        choices=("hybrid", "resnet18", "vit_small"),
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """저장된 체크포인트에 대해 전체 테스트 평가를 수행한다."""
    args = parse_args()
    cfg = load_config(args.config)
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    device = get_device(args.device)
    data_root = Path(__file__).resolve().parents[1] / "data" / "cifar_10"

    _, _, test_loader = build_loaders(
        root=data_root,
        image_size=cfg["image_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        seed=cfg["seed"],
        limit_train_samples=None,
        limit_val_samples=None,
        limit_test_samples=None,
    )

    model = build_model(args.model, cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    result = evaluate(
        model=model,
        loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        num_classes=cfg["num_classes"],
    )

    print(f"[INFO] checkpoint={args.checkpoint}")
    print(f"[INFO] device={device}")
    print(f"[TEST] loss={result['loss']:.4f} acc={result['acc']:.4f} macro_f1={result['macro_f1']:.4f}")
    print("[CONFUSION_MATRIX]")
    for row in result["confusion_matrix"].tolist():
        print(" ".join(str(value) for value in row))


if __name__ == "__main__":
    main()
