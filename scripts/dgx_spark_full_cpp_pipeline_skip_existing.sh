#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

can_run_python() {
  local candidate="$1"
  [[ -n "${candidate}" ]] || return 1
  [[ -x "${candidate}" ]] || return 1
  "${candidate}" -c 'import sys' >/dev/null 2>&1
}

pick_python() {
  if [[ -n "${PYTHON_BIN:-}" ]] && can_run_python "${PYTHON_BIN}"; then
    printf '%s\n' "${PYTHON_BIN}"
    return
  fi
  if can_run_python "${ROOT_DIR}/.venv/bin/python"; then
    printf '%s\n' "${ROOT_DIR}/.venv/bin/python"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    local python3_bin
    python3_bin="$(command -v python3)"
    if can_run_python "${python3_bin}"; then
      printf '%s\n' "${python3_bin}"
      return
    fi
  fi
  if command -v python >/dev/null 2>&1; then
    local python_bin
    python_bin="$(command -v python)"
    if can_run_python "${python_bin}"; then
      printf '%s\n' "${python_bin}"
      return
    fi
  fi
  printf '%s\n' "python executable not found" >&2
  exit 1
}

PYTHON_BIN="$(pick_python)"
MODEL_NAME="${MODEL_NAME:-hybrid}"
DEVICE="${DEVICE:-auto}"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/cifar10_hybrid_full.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/runs/dgx_spark_full_cpp}"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build_dgx_spark}"
CPP_WORK_DIR="${OUTPUT_DIR}/cpp_full_test"
INPUT_DIR="${CPP_WORK_DIR}/inputs"
MANIFEST_PATH="${INPUT_DIR}/manifest.tsv"
PREDICTIONS_TSV="${CPP_WORK_DIR}/predictions.tsv"
CPP_LOG="${CPP_WORK_DIR}/cpp_infer.log"
SUMMARY_PATH="${CPP_WORK_DIR}/summary.txt"
REPORT_PATH="${REPORT_PATH:-${ROOT_DIR}/prediction_results/cpp_full_test_report.html}"
TORCHSCRIPT_PATH="${OUTPUT_DIR}/hybrid_cifar10_full_ts.pt"
TRAIN_LOG="${OUTPUT_DIR}/train.log"
EVAL_LOG="${OUTPUT_DIR}/eval.log"
CHECKPOINT_PATH="${OUTPUT_DIR}/best.pth"
FINAL_CHECKPOINT_PATH="${OUTPUT_DIR}/final_${MODEL_NAME}.pth"
INFER_BIN="${BUILD_DIR}/cifar10_infer"
RUN_CPP_STAGE="${RUN_CPP_STAGE:-0}"

TRAIN_ARGS=(
  --config "${CONFIG_PATH}"
  --model "${MODEL_NAME}"
  --device "${DEVICE}"
  --output-dir "${OUTPUT_DIR}"
)
EVAL_ARGS=(
  --config "${CONFIG_PATH}"
  --checkpoint "${CHECKPOINT_PATH}"
  --model "${MODEL_NAME}"
  --device "${DEVICE}"
)

if [[ -n "${BATCH_SIZE_OVERRIDE:-}" ]]; then
  TRAIN_ARGS+=(--batch-size "${BATCH_SIZE_OVERRIDE}")
  EVAL_ARGS+=(--batch-size "${BATCH_SIZE_OVERRIDE}")
fi

if [[ -n "${NUM_WORKERS_OVERRIDE:-}" ]]; then
  TRAIN_ARGS+=(--num-workers "${NUM_WORKERS_OVERRIDE}")
  EVAL_ARGS+=(--num-workers "${NUM_WORKERS_OVERRIDE}")
fi

mkdir -p "${OUTPUT_DIR}" "${CPP_WORK_DIR}" "${INPUT_DIR}"

print_rule() {
  printf '%s\n' "============================================================"
}

print_section() {
  print_rule
  printf '[SECTION] %s\n' "$1"
  print_rule
}

require_python_modules() {
  local missing
  missing="$("${PYTHON_BIN}" - <<'PY'
import importlib.util

required = ("torch", "torchvision", "timm", "yaml", "tqdm", "numpy")
missing = [name for name in required if importlib.util.find_spec(name) is None]
print(" ".join(missing))
PY
)"
  if [[ -n "${missing}" ]]; then
    print_rule >&2
    printf '[ERROR] Missing Python modules for %s: %s\n' "${PYTHON_BIN}" "${missing}" >&2
    printf '[ERROR] Rebuild the Linux venv and install requirements: python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt\n' >&2
    print_rule >&2
    exit 1
  fi
}

generate_visual_report() {
  if [[ ! -f "${PREDICTIONS_TSV}" || ! -f "${CPP_LOG}" ]]; then
    return 1
  fi

  print_section "Visualization Report"
  "${PYTHON_BIN}" "${ROOT_DIR}/prediction_results/visualize_cpp_results.py" \
    --predictions "${PREDICTIONS_TSV}" \
    --log "${CPP_LOG}" \
    --run-dir "${OUTPUT_DIR}" \
    --output "${REPORT_PATH}"
}

select_checkpoint_path() {
  if [[ -f "${CHECKPOINT_PATH}" ]]; then
    printf '%s\n' "${CHECKPOINT_PATH}"
    return
  fi
  if [[ -f "${FINAL_CHECKPOINT_PATH}" ]]; then
    printf '%s\n' "${FINAL_CHECKPOINT_PATH}"
    return
  fi
  return 1
}

require_python_modules

print_section "Run Summary"
"${PYTHON_BIN}" - <<PY
import math
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("${CONFIG_PATH}").read_text(encoding="utf-8"))
batch_size = int("${BATCH_SIZE_OVERRIDE:-0}") or cfg["batch_size"]
num_workers = int("${NUM_WORKERS_OVERRIDE:-0}") or cfg["num_workers"]
train_samples = 45000
val_samples = 5000
test_samples = 10000
phase1 = cfg["epochs_phase1"]
phase2 = cfg["epochs_phase2"]
total_epochs = phase1 + phase2

print(f"python=${PYTHON_BIN}")
print(f"config={Path('${CONFIG_PATH}').name}")
print(f"device=${DEVICE}")
print(f"model=${MODEL_NAME}")
print(f"train_samples={train_samples}")
print(f"val_samples={val_samples}")
print(f"test_samples={test_samples}")
print(f"batch_size={batch_size}")
print(f"num_workers={num_workers}")
print(f"phase1_epochs={phase1}")
print(f"phase2_epochs={phase2}")
print(f"total_epochs={total_epochs}")
print(f"train_steps_per_epoch={math.ceil(train_samples / batch_size)}")
print(f"val_steps_per_epoch={math.ceil(val_samples / batch_size)}")
print(f"test_steps={math.ceil(test_samples / batch_size)}")
print(f"output_dir=${OUTPUT_DIR}")
PY

SELECTED_CHECKPOINT_PATH=""
SKIP_TRAINING=0
SKIP_EVALUATION=0
SKIP_EXPORT=0

if [[ -f "${TORCHSCRIPT_PATH}" ]]; then
  SKIP_TRAINING=1
  SKIP_EVALUATION=1
  SKIP_EXPORT=1
  SELECTED_CHECKPOINT_PATH="$(select_checkpoint_path || true)"
elif SELECTED_CHECKPOINT_PATH="$(select_checkpoint_path)"; then
  SKIP_TRAINING=1
  EVAL_ARGS=(
    --config "${CONFIG_PATH}"
    --checkpoint "${SELECTED_CHECKPOINT_PATH}"
    --model "${MODEL_NAME}"
    --device "${DEVICE}"
  )
  if [[ -n "${BATCH_SIZE_OVERRIDE:-}" ]]; then
    EVAL_ARGS+=(--batch-size "${BATCH_SIZE_OVERRIDE}")
  fi
  if [[ -n "${NUM_WORKERS_OVERRIDE:-}" ]]; then
    EVAL_ARGS+=(--num-workers "${NUM_WORKERS_OVERRIDE}")
  fi
fi

if [[ "${SKIP_TRAINING}" == "1" ]]; then
  print_section "Training"
  printf '[INFO] Existing artifact detected. Skipping training.\n'
  if [[ -n "${SELECTED_CHECKPOINT_PATH}" ]]; then
    printf '[INFO] checkpoint=%s\n' "${SELECTED_CHECKPOINT_PATH}"
  fi
  if [[ -f "${TORCHSCRIPT_PATH}" ]]; then
    printf '[INFO] torchscript=%s\n' "${TORCHSCRIPT_PATH}"
  fi
else
  print_section "Training"
  "${PYTHON_BIN}" "${ROOT_DIR}/python/train.py" "${TRAIN_ARGS[@]}" | tee "${TRAIN_LOG}"
  SELECTED_CHECKPOINT_PATH="$(select_checkpoint_path)"
  EVAL_ARGS=(
    --config "${CONFIG_PATH}"
    --checkpoint "${SELECTED_CHECKPOINT_PATH}"
    --model "${MODEL_NAME}"
    --device "${DEVICE}"
  )
  if [[ -n "${BATCH_SIZE_OVERRIDE:-}" ]]; then
    EVAL_ARGS+=(--batch-size "${BATCH_SIZE_OVERRIDE}")
  fi
  if [[ -n "${NUM_WORKERS_OVERRIDE:-}" ]]; then
    EVAL_ARGS+=(--num-workers "${NUM_WORKERS_OVERRIDE}")
  fi
fi

if [[ "${SKIP_EVALUATION}" == "1" ]]; then
  print_section "Evaluation"
  printf '[INFO] Existing TorchScript detected. Skipping evaluation.\n'
else
  print_section "Evaluation"
  "${PYTHON_BIN}" "${ROOT_DIR}/python/evaluate_checkpoint.py" "${EVAL_ARGS[@]}" | tee "${EVAL_LOG}"
fi

if [[ "${SKIP_EXPORT}" == "1" ]]; then
  print_section "TorchScript Export"
  printf '[INFO] Existing TorchScript detected. Skipping export.\n'
  printf '[INFO] torchscript=%s\n' "${TORCHSCRIPT_PATH}"
else
  print_section "TorchScript Export"
  "${PYTHON_BIN}" "${ROOT_DIR}/python/export_torchscript.py" \
    --config "${CONFIG_PATH}" \
    --checkpoint "${SELECTED_CHECKPOINT_PATH}" \
    --model "${MODEL_NAME}" \
    --output "${TORCHSCRIPT_PATH}" \
    --device cpu
fi

print_section "Export CIFAR-10 Test Images"
"${PYTHON_BIN}" - <<PY
from pathlib import Path
from torchvision import datasets

classes = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

project_root = Path("${ROOT_DIR}")
data_root = project_root / "data" / "cifar_10"
input_dir = Path("${INPUT_DIR}")
input_dir.mkdir(parents=True, exist_ok=True)
manifest_path = Path("${MANIFEST_PATH}")
dataset = datasets.CIFAR10(root=str(data_root), train=False, download=True)

lines = ["order\tdataset_index\ttarget_idx\ttarget_name\tfile_name"]
for dataset_index, (image, target_idx) in enumerate(dataset):
    target_name = classes[target_idx]
    file_name = f"{dataset_index:05d}_{target_name}.png"
    image.save(input_dir / file_name)
    lines.append(
        f"{dataset_index + 1}\t{dataset_index}\t{target_idx}\t{target_name}\t{file_name}"
    )

manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"saved={len(dataset)}")
print(f"manifest={manifest_path}")
PY

if [[ "${RUN_CPP_STAGE}" != "1" ]]; then
  generate_visual_report || true

  print_section "Manual Next Steps"
  printf '%s\n' "Python artifacts are ready. C++ build/inference was skipped."
  if [[ -n "${SELECTED_CHECKPOINT_PATH}" ]]; then
    printf '[INFO] checkpoint=%s\n' "${SELECTED_CHECKPOINT_PATH}"
  fi
  printf '[INFO] torchscript=%s\n' "${TORCHSCRIPT_PATH}"
  printf '[INFO] manifest=%s\n' "${MANIFEST_PATH}"
  if [[ -f "${REPORT_PATH}" ]]; then
    printf '[INFO] report=%s\n' "${REPORT_PATH}"
  fi
  printf '%s\n' ""
  printf '%s\n' "To continue manually after installing OpenCV development files:"
  printf '  sudo apt-get install -y libopencv-dev\n'
  printf '  TORCH_CMAKE_PREFIX_PATH="%s" cmake -S "%s" -B "%s" -DCMAKE_PREFIX_PATH="%s"\n' \
    "${LIBTORCH_CMAKE_PREFIX_PATH:-$("${PYTHON_BIN}" -c 'import torch; print(torch.utils.cmake_prefix_path)')}" \
    "${ROOT_DIR}/cpp_infer" \
    "${BUILD_DIR}" \
    "${LIBTORCH_CMAKE_PREFIX_PATH:-$("${PYTHON_BIN}" -c 'import torch; print(torch.utils.cmake_prefix_path)')}"
  printf '  cmake --build "%s" --config Release\n' "${BUILD_DIR}"
  printf '  "%s" "%s" "%s" "%s" | tee "%s"\n' \
    "${INFER_BIN}" "${TORCHSCRIPT_PATH}" "${MANIFEST_PATH}" "${PREDICTIONS_TSV}" "${CPP_LOG}"
  printf '%s\n' ""
  printf '%s\n' "Set RUN_CPP_STAGE=1 if you want this script to run the C++ stage automatically."
  exit 0
fi

print_section "Build C++ Inference"
TORCH_CMAKE_PREFIX_PATH="${LIBTORCH_CMAKE_PREFIX_PATH:-$("${PYTHON_BIN}" -c 'import torch; print(torch.utils.cmake_prefix_path)')}"
cmake -S "${ROOT_DIR}/cpp_infer" -B "${BUILD_DIR}" -DCMAKE_PREFIX_PATH="${TORCH_CMAKE_PREFIX_PATH}"
cmake --build "${BUILD_DIR}" --config Release

print_section "C++ Full Test Inference"
"${INFER_BIN}" "${TORCHSCRIPT_PATH}" "${MANIFEST_PATH}" "${PREDICTIONS_TSV}" | tee "${CPP_LOG}"

generate_visual_report

print_section "Build Summary"
"${PYTHON_BIN}" - <<PY
import csv
from collections import Counter
from pathlib import Path

classes = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

predictions_path = Path("${PREDICTIONS_TSV}")
summary_path = Path("${SUMMARY_PATH}")
rows = list(csv.DictReader(predictions_path.open("r", encoding="utf-8"), delimiter="\t"))
if not rows:
    raise RuntimeError(f"No prediction rows found in {predictions_path}")

total = len(rows)
correct = sum(int(row["correct"]) for row in rows)
accuracy = correct / total
pred_counts = Counter(row["pred_name"] for row in rows)
target_counts = Counter(row["target_name"] for row in rows)
confusion = [[0 for _ in classes] for _ in classes]
for row in rows:
    target_idx = int(row["target_idx"])
    pred_idx = int(row["pred_idx"])
    confusion[target_idx][pred_idx] += 1

lines = [
    f"config={Path('${CONFIG_PATH}').name}",
    f"device=${DEVICE}",
    f"model=${MODEL_NAME}",
    f"checkpoint=${SELECTED_CHECKPOINT_PATH}",
    f"final_checkpoint=${FINAL_CHECKPOINT_PATH}",
    f"torchscript=${TORCHSCRIPT_PATH}",
    f"manifest=${MANIFEST_PATH}",
    f"predictions=${PREDICTIONS_TSV}",
    f"samples={total}",
    f"correct={correct}",
    f"accuracy={accuracy:.4f}",
    "",
    "target_counts:",
]
for class_name in classes:
    lines.append(f"{class_name}\t{target_counts[class_name]}")

lines.append("")
lines.append("prediction_counts:")
for class_name, count in sorted(pred_counts.items(), key=lambda item: (-item[1], item[0])):
    lines.append(f"{class_name}\t{count}")

lines.append("")
lines.append("confusion_matrix_header:")
lines.append("\t" + "\t".join(classes))
for class_name, row in zip(classes, confusion):
    lines.append(class_name + "\t" + "\t".join(str(value) for value in row))

summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_path.read_text(encoding="utf-8"), end="")
PY

print_section "Artifacts"
printf 'train_log=%s\n' "${TRAIN_LOG}"
printf 'eval_log=%s\n' "${EVAL_LOG}"
printf 'checkpoint=%s\n' "${CHECKPOINT_PATH}"
printf 'final_checkpoint=%s\n' "${FINAL_CHECKPOINT_PATH}"
printf 'torchscript=%s\n' "${TORCHSCRIPT_PATH}"
printf 'cpp_predictions=%s\n' "${PREDICTIONS_TSV}"
printf 'cpp_log=%s\n' "${CPP_LOG}"
printf 'cpp_summary=%s\n' "${SUMMARY_PATH}"
printf 'cpp_report=%s\n' "${REPORT_PATH}"
