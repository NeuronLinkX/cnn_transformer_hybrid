"""재현성 설정 헬퍼."""

# 타입 힌트의 지연 평가를 사용한다.
from __future__ import annotations

# PYTHONHASHSEED 제어를 위해 os를 가져온다.
import os
# 파이썬 수준 RNG 제어를 위해 random을 가져온다.
import random

# NumPy RNG 제어를 위해 NumPy를 가져온다.
import numpy as np
# PyTorch RNG와 백엔드 설정을 위해 torch를 가져온다.
import torch


def set_seed(seed: int, deterministic: bool = False):
    """Python, NumPy, PyTorch 시드를 설정한다."""
    # 파이썬 기본 random 모듈의 시드를 설정한다.
    random.seed(seed)
    # NumPy 난수 생성기의 시드를 설정한다.
    np.random.seed(seed)
    # 현재 프로세스의 해시 기반 무작위성을 고정한다.
    os.environ["PYTHONHASHSEED"] = str(seed)
    # CPU 측 PyTorch 난수 생성기의 시드를 설정한다.
    torch.manual_seed(seed)

    # CUDA가 실제로 사용 가능할 때만 CUDA 난수 생성기 시드를 설정한다.
    if torch.cuda.is_available():
        # 현재 CUDA 장치의 난수 생성기 시드를 설정한다.
        torch.cuda.manual_seed(seed)
        # 멀티 GPU 환경을 위해 모든 CUDA 장치 시드를 설정한다.
        torch.cuda.manual_seed_all(seed)

    # deterministic=True는 재현성을 높이지만 실행 속도를 떨어뜨릴 수 있다.
    torch.backends.cudnn.deterministic = deterministic
    # 결정적 커널이 필요할 때는 cuDNN 벤치마킹을 끈다.
    torch.backends.cudnn.benchmark = not deterministic
