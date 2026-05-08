# Full Training And C++ Results Guide

이 문서는 Linux/DGX Spark 환경에서 아래 두 작업을 끝까지 수행하는 절차를 정리한다.

- Python으로 CIFAR-10 전체 학습
- 현재 저장소의 C++ 추론 바이너리로 전체 테스트셋 결과 생성

작업 디렉터리는 프로젝트 루트 기준이다.

```bash
cd /home/jiwoo/Desktop/workspace/cnn_transformer_cifar10
```

## 1. 목적과 산출물

최종적으로 아래 파일들이 생성되면 전체 파이프라인이 완료된 것이다.

- `runs/dgx_spark_full_cpp/best.pth`
- `runs/dgx_spark_full_cpp/final_hybrid.pth`
- `runs/dgx_spark_full_cpp/train.log`
- `runs/dgx_spark_full_cpp/eval.log`
- `runs/dgx_spark_full_cpp/hybrid_cifar10_full_ts.pt`
- `runs/dgx_spark_full_cpp/cpp_full_test/inputs/manifest.tsv`
- `runs/dgx_spark_full_cpp/cpp_full_test/predictions.tsv`
- `runs/dgx_spark_full_cpp/cpp_full_test/cpp_infer.log`
- `runs/dgx_spark_full_cpp/cpp_full_test/summary.txt`

## 2. Python 환경 준비

Ubuntu 계열에서 `venv`가 없으면 먼저 설치한다.

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv
```

새 Linux 가상환경을 만들고 의존성을 설치한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

필수 패키지 확인:

```bash
python -c "import torch, torchvision, timm, yaml, tqdm, numpy; print(torch.__version__)"
```

GPU 확인:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

## 3. 간단한 실행 방법

학습, 평가, TorchScript export, 테스트 이미지 export까지 자동 실행하는 스크립트는 아래다.

```bash
./scripts/dgx_spark_full_cpp_pipeline.sh
```

기본 설정에서 이 스크립트는 순서대로 아래를 수행한다.

- 전체 학습
- 전체 테스트셋 평가
- TorchScript export
- CIFAR-10 테스트 10,000장 PNG export

기본값에서는 여기서 종료한다. `OpenCV`가 준비된 환경에서만 C++ 단계는 수동으로 진행한다.

자동으로 C++ 단계까지 이어서 실행하려면 아래처럼 `RUN_CPP_STAGE=1`을 준다.

```bash
RUN_CPP_STAGE=1 ./scripts/dgx_spark_full_cpp_pipeline.sh
```

실행 중 맨 처음 출력되는 `python=...` 값이 실제 사용 인터프리터다. 현재 스크립트는 깨진 `.venv`를 자동으로 건너뛰고, `torch`, `torchvision`, `timm` 등이 빠져 있으면 학습 전에 바로 실패하도록 되어 있다.

## 4. C++ 단계 수동 실행

현재 에러는 `OpenCVConfig.cmake`를 찾지 못해서 발생한 것이다. Ubuntu 계열이면 먼저 개발 패키지를 설치한다.

```bash
sudo apt-get install -y libopencv-dev
```

그 다음 C++ 추론 바이너리를 직접 빌드한다.

```bash
TORCH_CMAKE_PREFIX_PATH="$(.venv/bin/python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake -S cpp_infer -B build_dgx_spark -DCMAKE_PREFIX_PATH="${TORCH_CMAKE_PREFIX_PATH}"

cmake --build build_dgx_spark --config Release
```

이 프로젝트의 `cifar10_infer`는 단일 이미지 모드와 manifest 기반 배치 모드를 모두 지원한다. 전체 테스트셋 결과를 다시 만들려면 아래를 실행한다.

```bash
build_dgx_spark/cifar10_infer \
  runs/dgx_spark_full_cpp/hybrid_cifar10_full_ts.pt \
  runs/dgx_spark_full_cpp/cpp_full_test/inputs/manifest.tsv \
  runs/dgx_spark_full_cpp/cpp_full_test/predictions.tsv | \
  tee runs/dgx_spark_full_cpp/cpp_full_test/cpp_infer.log
```

배치 추론 후 요약 파일은 필요하면 수동으로 추가 생성해야 한다. 현재 자동 스크립트에서 요약 생성은 `RUN_CPP_STAGE=1`일 때만 이어서 수행된다.
