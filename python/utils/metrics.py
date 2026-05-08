"""평가 지표와 평가 루프."""

# 타입 힌트의 지연 평가를 사용한다.
from __future__ import annotations

# 텐서 연산과 no-grad 평가를 위해 torch를 가져온다.
import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """가장 큰 점수 기준의 분류 정확도를 계산한다."""
    # 각 샘플에 대해 가장 큰 로짓의 클래스를 선택한다.
    pred = logits.argmax(dim=1)
    # 일치 비율의 평균을 파이썬 float로 반환한다.
    return (pred == targets).float().mean().item()


def confusion_matrix(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """로짓과 정수 타깃으로부터 단순 혼동 행렬을 계산한다."""
    # 로짓을 이산 클래스 예측값으로 변환한다.
    pred = logits.argmax(dim=1)
    # 로그 출력을 쉽게 하기 위해 CPU 위에 0으로 초기화된 혼동 행렬을 만든다.
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    # 모든 정답/예측 쌍을 행렬에 누적한다.
    for true_idx, pred_idx in zip(targets.view(-1).cpu(), pred.view(-1).cpu()):
        matrix[true_idx.long(), pred_idx.long()] += 1
    # 채워진 혼동 행렬을 반환한다.
    return matrix


def macro_f1_from_confusion(matrix: torch.Tensor) -> float:
    """추가 의존성 없이 혼동 행렬로부터 macro F1을 계산한다."""
    # 나눗셈을 위해 count를 float로 변환한다.
    matrix = matrix.to(torch.float32)
    # 대각선에서 true positive를 추출한다.
    tp = matrix.diag()
    # 클래스별 예측 양성 수를 합산한다.
    pred_pos = matrix.sum(dim=0)
    # 클래스별 실제 양성 수를 합산한다.
    true_pos = matrix.sum(dim=1)
    # 0으로 나누는 상황을 방지하며 precision을 계산한다.
    precision = tp / pred_pos.clamp_min(1.0)
    # 0으로 나누는 상황을 방지하며 recall을 계산한다.
    recall = tp / true_pos.clamp_min(1.0)
    # 0으로 나누는 상황을 방지하며 클래스별 F1을 계산한다.
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    # 클래스 전반의 macro 평균 F1을 반환한다.
    return f1.mean().item()


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int | None = None):
    """평균 손실과 정확도를 반환하고 필요하면 macro F1과 혼동 행렬도 계산한다."""
    # 모델을 평가 모드로 전환한다.
    model.eval()
    # 전체 로더 순회를 위한 스칼라 누적값을 초기화한다.
    total_loss, total_correct, total = 0.0, 0, 0
    # 선택 지표를 사용하지 않을 때는 혼동 행렬을 비워 둔다.
    matrix = None

    # gradient 추적 없이 평가 배치를 순회한다.
    for x, y in loader:
        # 이미지를 대상 장치로 이동한다.
        x = x.to(device, non_blocking=True)
        # 레이블을 대상 장치로 이동한다.
        y = y.to(device, non_blocking=True)
        # 모델 순전파를 수행한다.
        logits = model(x)
        # 배치 손실을 계산한다.
        loss = criterion(logits, y)
        # 배치 크기를 반영하여 손실을 누적한다.
        total_loss += loss.item() * x.size(0)
        # 정답 예측 개수를 누적한다.
        total_correct += (logits.argmax(1) == y).sum().item()
        # 처리한 샘플 수를 누적한다.
        total += x.size(0)

        # 요청된 경우 선택적 혼동 행렬 통계를 계산한다.
        if num_classes is not None:
            # 현재 배치의 혼동 행렬을 CPU에서 계산한다.
            batch_matrix = confusion_matrix(logits.detach(), y.detach(), num_classes)
            # 첫 배치에서는 누적 행렬을 초기화한다.
            matrix = batch_matrix if matrix is None else matrix + batch_matrix

    # 핵심 평가 지표를 기존 코드와 같은 dict 형식으로 묶는다.
    result = {
        "loss": total_loss / max(total, 1),
        "acc": total_correct / max(total, 1),
    }

    # 요청된 경우 혼동 행렬 기반 추가 지표를 붙인다.
    if matrix is not None:
        # 후속 보고를 위해 혼동 행렬 자체를 저장한다.
        result["confusion_matrix"] = matrix
        # 파생된 macro F1 점수를 저장한다.
        result["macro_f1"] = macro_f1_from_confusion(matrix)

    # 완성된 지표 딕셔너리를 반환한다.
    return result
