"""CIFAR-10 데이터 유틸리티."""

# 타입 힌트의 지연 평가를 사용한다.
from __future__ import annotations

# 로더가 파일시스템 경로를 깔끔하게 받을 수 있도록 Path를 가져온다.
from pathlib import Path

# 재현 가능한 인덱스 분할을 위해 torch를 가져온다.
import torch
# 배치 처리와 분할 처리를 위해 DataLoader와 Subset을 가져온다.
from torch.utils.data import DataLoader, Subset
# CIFAR-10 데이터셋 클래스와 변환 연산을 가져온다.
from torchvision import datasets, transforms


# 정규화에 사용할 CIFAR-10 표준 채널 평균값이다.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
# 정규화에 사용할 CIFAR-10 표준 채널 표준편차이다.
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_loaders(
    # 데이터셋 루트는 문자열 경로 또는 Path 객체를 받을 수 있다.
    root: str | Path = "./data",
    # CIFAR-10 이미지를 요청된 정사각 해상도로 리사이즈한다.
    image_size: int = 224,
    # 모든 로더에 사용할 배치 크기이다.
    batch_size: int = 64,
    # DataLoader 워커 수이다.
    num_workers: int = 4,
    # 학습/검증 분할에 사용할 랜덤 시드이다.
    seed: int = 42,
    # 빠른 스모크 테스트를 위해 학습 샘플 수를 제한할 수 있다.
    limit_train_samples: int | None = None,
    # 빠른 스모크 테스트를 위해 검증 샘플 수를 제한할 수 있다.
    limit_val_samples: int | None = None,
    # 빠른 스모크 테스트를 위해 테스트 샘플 수를 제한할 수 있다.
    limit_test_samples: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """CIFAR-10 학습/검증/테스트 로더를 생성한다."""
    # 일관된 처리를 위해 입력 루트를 Path 객체로 변환한다.
    root = Path(root)
    # torchvision이 파일을 내려받기 전에 데이터셋 디렉터리가 존재하도록 보장한다.
    root.mkdir(parents=True, exist_ok=True)

    # 설계에 맞춰 먼저 리사이즈한 뒤 crop 기반 증강을 적용한다.
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=16),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    # 검증과 테스트 전처리는 결정적으로 유지한다.
    test_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    # 증강이 포함된 학습 데이터셋을 생성한다.
    full_train_aug = datasets.CIFAR10(
        root=str(root),
        train=True,
        download=True,
        transform=train_tf,
    )
    # 검증 전용으로 증강이 없는 두 번째 학습 데이터셋을 생성한다.
    full_train_eval = datasets.CIFAR10(
        root=str(root),
        train=True,
        download=True,
        transform=test_tf,
    )
    # 홀드아웃 CIFAR-10 테스트 데이터셋을 생성한다.
    test_set = datasets.CIFAR10(
        root=str(root),
        train=False,
        download=True,
        transform=test_tf,
    )

    # 90/10 비율의 학습/검증 분할 크기를 계산한다.
    train_len = int(len(full_train_aug) * 0.9)
    # 남은 샘플은 검증용으로 사용한다.
    val_len = len(full_train_aug) - train_len
    # 재현 가능한 분할을 위해 전용 torch generator에 시드를 설정한다.
    generator = torch.Generator().manual_seed(seed)
    # 데이터셋 인덱스의 고정 순열을 생성한다.
    indices = torch.randperm(len(full_train_aug), generator=generator).tolist()
    # 앞부분은 학습 인덱스로 사용한다.
    train_indices = indices[:train_len]
    # 나머지 부분은 검증 인덱스로 사용한다.
    val_indices = indices[train_len:]

    # 증강된 데이터셋에 학습 인덱스를 적용한다.
    train_set = Subset(full_train_aug, train_indices)
    # 증강이 없는 데이터셋에 검증 인덱스를 적용한다.
    val_set = Subset(full_train_eval, val_indices)

    # 스모크 테스트용 샘플 제한이 있으면 학습 분할을 줄인다.
    if limit_train_samples is not None:
        # 앞에서부터 결정적으로 N개 학습 인덱스만 유지한다.
        train_set = Subset(train_set, list(range(min(limit_train_samples, len(train_set)))))

    # 스모크 테스트용 샘플 제한이 있으면 검증 분할을 줄인다.
    if limit_val_samples is not None:
        # 앞에서부터 결정적으로 N개 검증 인덱스만 유지한다.
        val_set = Subset(val_set, list(range(min(limit_val_samples, len(val_set)))))

    # 스모크 테스트용 샘플 제한이 있으면 테스트 분할을 줄인다.
    if limit_test_samples is not None:
        # 앞에서부터 결정적으로 N개 테스트 인덱스만 유지한다.
        test_set = Subset(test_set, list(range(min(limit_test_samples, len(test_set)))))

    # CUDA 사용 가능 시 pinned memory를 활성화한다.
    pin_memory = torch.cuda.is_available()
    # 워커 프로세스를 사용할 때만 persistent worker를 유지한다.
    persistent_workers = num_workers > 0

    # 셔플되는 학습 로더를 생성한다.
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    # 결정적인 검증 로더를 생성한다.
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    # 결정적인 테스트 로더를 생성한다.
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    # 표준 3개 분할 로더를 반환한다.
    return train_loader, val_loader, test_loader


def create_cifar10_loaders(
    data_dir: str | Path,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
    limit_train_samples: int | None = None,
    limit_val_samples: int | None = None,
    limit_test_samples: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """프로젝트 내부의 기존 import를 위한 호환 래퍼이다."""
    # 두 함수 이름이 동일하게 동작하도록 메인 로더 생성기를 재사용한다.
    return build_loaders(
        root=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        limit_train_samples=limit_train_samples,
        limit_val_samples=limit_val_samples,
        limit_test_samples=limit_test_samples,
    )
