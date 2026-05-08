"""CIFAR-10 하이브리드 모델과 베이스라인 모델의 학습 진입점."""

# 타입 힌트의 지연 평가를 사용한다.
from __future__ import annotations

# 명령행 인자 처리를 위해 argparse를 가져온다.
import argparse
# 출력 경로와 설정 파일 경로 처리를 위해 Path를 가져온다.
from pathlib import Path
# 설정 딕셔너리 타입 표기를 위해 Any를 가져온다.
from typing import Any

# PyTorch 기본 패키지를 가져온다.
import torch
# 신경망 구성 요소를 가져온다.
from torch import nn
# 요구된 옵티마이저인 AdamW를 가져온다.
from torch.optim import AdamW
# 코사인 어닐링 학습률 스케줄러를 가져온다.
from torch.optim.lr_scheduler import CosineAnnealingLR
# 배치 진행률 표시를 위해 tqdm을 가져온다.
from tqdm import tqdm
# 설정 파일 로딩을 위해 yaml을 가져온다.
import yaml

# 베이스라인 모델 생성 함수를 가져온다.
from models.baselines import create_resnet18, create_vit_small
# 하이브리드 CNN-Transformer 모델을 가져온다.
from models.hybrid import CNNTransformerHybrid
# CIFAR-10 DataLoader 생성 함수를 가져온다.
from utils.data import build_loaders
# 평가 헬퍼를 가져온다.
from utils.metrics import evaluate
# 재현성 설정 헬퍼를 가져온다.
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """명령행 인자를 파싱한다."""
    # 최상위 인자 파서를 생성한다.
    parser = argparse.ArgumentParser(description="Train CIFAR-10 classification models.")
    # YAML 설정 파일 경로 인자를 추가한다.
    parser.add_argument("--config", type=str, default="configs/cifar10_hybrid_smoke.yaml")
    # 장치 선택 인자를 추가한다.
    parser.add_argument("--device", type=str, default="auto")
    # 베이스라인 지원을 위해 모델 종류 선택 인자를 추가한다.
    parser.add_argument(
        "--model",
        type=str,
        default="hybrid",
        choices=("hybrid", "resnet18", "vit_small"),
    )
    # 체크포인트를 실행별로 묶기 위한 출력 디렉터리 인자를 추가한다.
    parser.add_argument("--output-dir", type=str, default="runs/smoke_test")
    # 필요하면 CLI에서 배치 크기를 덮어쓸 수 있게 한다.
    parser.add_argument("--batch-size", type=int, default=None)
    # 필요하면 CLI에서 DataLoader 워커 수를 덮어쓸 수 있게 한다.
    parser.add_argument("--num-workers", type=int, default=None)
    # 파싱된 명령행 인자를 반환한다.
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    """YAML 설정 파일을 딕셔너리로 불러온다."""
    # YAML 파일을 UTF-8 인코딩으로 연다.
    with open(path, "r", encoding="utf-8") as fp:
        # YAML 문서를 파이썬 객체로 파싱한다.
        return yaml.safe_load(fp)


def get_device(name: str = "auto") -> torch.device:
    """'auto' 또는 명시적 장치 문자열을 torch.device로 변환한다."""
    # 사용자가 auto를 요청했으면 CUDA를 우선 선택한다.
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    # 그렇지 않으면 요청된 장치를 그대로 생성한다.
    return torch.device(name)


def build_model(model_name: str, cfg: dict[str, Any]) -> nn.Module:
    """설정에 따라 선택된 모델 계열을 생성한다."""
    # 중첩된 모델 설정을 한 번만 읽어 접근을 단순화한다.
    mcfg = cfg["model"]

    # 요청된 경우 하이브리드 모델을 생성한다.
    if model_name == "hybrid":
        return CNNTransformerHybrid(
            backbone_name=mcfg["backbone"],
            num_classes=cfg["num_classes"],
            pretrained=mcfg["pretrained"],
            embed_dim=mcfg["embed_dim"],
            transformer_depth=mcfg["transformer_depth"],
            num_heads=mcfg["num_heads"],
            mlp_ratio=mcfg["mlp_ratio"],
            dropout=mcfg["dropout"],
        )

    # 요청된 경우 ResNet-18 분류기 베이스라인을 생성한다.
    if model_name == "resnet18":
        return create_resnet18(
            num_classes=cfg["num_classes"],
            pretrained=mcfg["pretrained"],
        )

    # 요청된 경우 ViT-small 분류기 베이스라인을 생성한다.
    if model_name == "vit_small":
        return create_vit_small(
            num_classes=cfg["num_classes"],
            pretrained=mcfg["pretrained"],
        )

    # 지원하지 않는 모델 이름이 들어오면 명시적으로 실패시킨다.
    raise ValueError(f"Unsupported model choice: {model_name}")


def run_epoch(model, loader, criterion, optimizer, device):
    """한 번의 학습 epoch를 수행하고 손실/정확도 통계를 반환한다."""
    # 모델을 학습 모드로 전환한다.
    model.train()
    # 누적 손실, 정답 수, 샘플 수를 초기화한다.
    total_loss, total_correct, total = 0.0, 0, 0

    # 간단한 진행률 표시와 함께 학습 배치를 순회한다.
    for x, y in tqdm(loader, leave=False):
        # 입력 이미지를 선택된 장치로 이동한다.
        x = x.to(device, non_blocking=True)
        # 레이블을 선택된 장치로 이동한다.
        y = y.to(device, non_blocking=True)
        # 새 역전파 전에 이전 gradient를 비운다.
        optimizer.zero_grad(set_to_none=True)
        # 순전파를 수행해 클래스 로짓을 얻는다.
        logits = model(x)
        # 현재 배치의 크로스엔트로피 손실을 계산한다.
        loss = criterion(logits, y)
        # 손실을 학습 가능한 파라미터에 대해 역전파한다.
        loss.backward()
        # 학습 안정화를 위해 gradient clipping을 적용한다.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # 옵티마이저 업데이트를 한 번 수행한다.
        optimizer.step()
        # 배치 크기를 반영해 손실을 누적한다.
        total_loss += loss.item() * x.size(0)
        # 정답 예측 개수를 누적한다.
        total_correct += (logits.argmax(1) == y).sum().item()
        # 처리한 샘플 수를 누적한다.
        total += x.size(0)

    # 평균 학습 지표를 기존 dict 형식으로 반환한다.
    return {"loss": total_loss / max(total, 1), "acc": total_correct / max(total, 1)}


def train_phase(model, train_loader, val_loader, device, epochs, lr, weight_decay, tag, output_dir, cfg):
    """단일 학습 단계를 수행하고 해당 단계의 최고 체크포인트를 저장한다."""
    # 분류에 사용하는 표준 크로스엔트로피 손실을 생성한다.
    criterion = nn.CrossEntropyLoss()
    # 현재 학습 가능한 파라미터만 최적화 대상으로 제한한다.
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    # 현재 단계 길이에 맞춰 코사인 어닐링을 적용한다.
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    # 현재 단계에서 최고 검증 정확도를 추적한다.
    best_acc = -1.0

    # 요청된 epoch 수만큼 현재 단계를 학습한다.
    for epoch in range(1, epochs + 1):
        # 학습 epoch를 한 번 수행한다.
        tr = run_epoch(model, train_loader, criterion, optimizer, device)
        # 검증 분할에서 평가한다.
        va = evaluate(model, val_loader, criterion, device, num_classes=cfg["num_classes"])
        # epoch 종료 후 학습률 스케줄을 한 단계 진행한다.
        scheduler.step()
        # 요청된 형식으로 단계 로그를 출력한다.
        print(
            f"[{tag}] epoch={epoch:03d} train_loss={tr['loss']:.4f} "
            f"train_acc={tr['acc']:.4f} val_loss={va['loss']:.4f} val_acc={va['acc']:.4f}"
        )

        # 현재 단계와 전체 실행 기준의 최고 체크포인트를 저장한다.
        if va["acc"] > best_acc:
            # 현재 단계 최고 검증 정확도를 갱신한다.
            best_acc = va["acc"]
            # 단순 state_dict 대신 추가 정보를 포함한 체크포인트를 구성한다.
            checkpoint = {
                "model_name": cfg.get("active_model", "hybrid"),
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "phase": tag,
                "val_acc": va["acc"],
            }
            # 단계별 최고 체크포인트를 저장해 확인을 쉽게 한다.
            torch.save(checkpoint, output_dir / f"best_{tag}.pth")
            # README에서 사용하는 대표 최고 체크포인트 경로도 함께 저장한다.
            torch.save(checkpoint, output_dir / "best.pth")

    # 현재 단계의 최고 검증 정확도를 반환한다.
    return best_acc


def main():
    """설정된 학습 워크플로를 실행한다."""
    # 먼저 명령행 인자를 파싱한다.
    args = parse_args()
    # YAML 설정 파일을 불러온다.
    cfg = load_config(args.config)
    # 체크포인트 export 편의를 위해 현재 모델 이름을 설정에 기록한다.
    cfg["active_model"] = args.model
    # CPU나 제한된 환경에서는 무거운 기본 설정을 CLI로 덮어쓸 수 있게 한다.
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    # Python, NumPy, PyTorch의 RNG 시드를 고정한다.
    set_seed(cfg["seed"], deterministic=False)
    # 실제 사용할 torch 장치를 결정한다.
    device = get_device(args.device)
    # 디버깅을 위해 선택된 장치를 출력한다.
    print(f"[INFO] device={device}")

    # 요청된 출력 디렉터리를 생성한다.
    output_dir = Path(args.output_dir)
    # 체크포인트 저장 전에 출력 디렉터리가 존재하도록 한다.
    output_dir.mkdir(parents=True, exist_ok=True)

    # 프로젝트 내부의 데이터셋 디렉터리를 결정한다.
    data_root = Path(__file__).resolve().parents[1] / "data" / "cifar_10"
    # 학습, 검증, 테스트 DataLoader를 생성한다.
    train_loader, val_loader, test_loader = build_loaders(
        root=data_root,
        image_size=cfg["image_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        seed=cfg["seed"],
        limit_train_samples=cfg.get("limit_train_samples"),
        limit_val_samples=cfg.get("limit_val_samples"),
        limit_test_samples=cfg.get("limit_test_samples"),
    )

    # 선택된 모델을 생성하고 대상 장치로 이동한다.
    model = build_model(args.model, cfg).to(device)

    # 하이브리드 모델은 2단계 학습 경로로 실행한다.
    if args.model == "hybrid":
        # 1단계에서는 CNN 백본을 고정하고 Transformer와 head만 학습한다.
        model.freeze_backbone(True)
        # 1단계 하이퍼파라미터로 학습을 수행한다.
        train_phase(
            model,
            train_loader,
            val_loader,
            device,
            cfg["epochs_phase1"],
            cfg["lr_phase1"],
            cfg["weight_decay"],
            "phase1",
            output_dir,
            cfg,
        )

        # 2단계에서는 전체 모델을 풀어 end-to-end 미세조정을 수행한다.
        model.freeze_backbone(False)
        # 두 번째 학습률로 2단계 학습을 수행한다.
        train_phase(
            model,
            train_loader,
            val_loader,
            device,
            cfg["epochs_phase2"],
            cfg["lr_phase2"],
            cfg["weight_decay"],
            "phase2",
            output_dir,
            cfg,
        )
    else:
        # 일반 베이스라인 모델은 단일 단계 학습으로 처리한다.
        train_phase(
            model,
            train_loader,
            val_loader,
            device,
            cfg["epochs_phase1"] + cfg["epochs_phase2"],
            cfg["lr_phase1"],
            cfg["weight_decay"],
            "train",
            output_dir,
            cfg,
        )

    # 최종 테스트 평가를 위해 손실 함수를 다시 생성한다.
    criterion = nn.CrossEntropyLoss()
    # 최고 체크포인트가 있으면 그것을 우선 사용한다.
    best_path = output_dir / "best.pth"

    # 테스트 결과를 출력하기 전에 최고 체크포인트를 다시 불러온다.
    if best_path.exists():
        # 체크포인트를 현재 장치로 불러온다.
        checkpoint = torch.load(best_path, map_location=device)
        # 실제 state_dict를 체크포인트 payload에서 추출한다.
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        # 최고 성능 모델 파라미터를 복원한다.
        model.load_state_dict(state_dict)

    # 테스트 분할에서 평가한다.
    test = evaluate(model, test_loader, criterion, device, num_classes=cfg["num_classes"])
    # 요청된 형식으로 최종 테스트 지표를 출력한다.
    print(f"[TEST] loss={test['loss']:.4f} acc={test['acc']:.4f}")

    # export 편의를 위해 최종 체크포인트 payload를 저장한다.
    torch.save(
        {
            "model_name": args.model,
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "phase": "final",
            "val_acc": None,
        },
        output_dir / f"final_{args.model}.pth",
    )


# 스크립트가 직접 실행될 때만 메인 루틴을 수행한다.
if __name__ == "__main__":
    # 학습 메인 워크플로에 진입한다.
    main()
