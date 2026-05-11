#!/usr/bin/env python3
"""Build a visual HTML report for the C++ CIFAR-10 batch inference outputs."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import mimetypes
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency at runtime
    torch = None

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency at runtime
    yaml = None


CLASS_NAMES = (
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

@dataclass
class PredictionRow:
    order: int
    dataset_index: int
    target_idx: int
    target_name: str
    pred_idx: int
    pred_name: str
    confidence: float
    correct: bool
    image_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("runs/dgx_spark_full_cpp/cpp_full_test/predictions.tsv"),
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("runs/dgx_spark_full_cpp/cpp_full_test/cpp_infer.log"),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/dgx_spark_full_cpp"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prediction_results/cpp_full_test_report.html"),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=24,
        help="Number of image cards to show per gallery.",
    )
    return parser.parse_args()


def load_predictions(path: Path) -> list[PredictionRow]:
    rows: list[PredictionRow] = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for raw in reader:
            rows.append(
                PredictionRow(
                    order=int(raw["order"]),
                    dataset_index=int(raw["dataset_index"]),
                    target_idx=int(raw["target_idx"]),
                    target_name=raw["target_name"],
                    pred_idx=int(raw["pred_idx"]),
                    pred_name=raw["pred_name"],
                    confidence=float(raw["confidence"]),
                    correct=raw["correct"] == "1",
                    image_path=Path(raw["image_path"]),
                )
            )
    if not rows:
        raise ValueError(f"No prediction rows found in {path}")
    return rows


def parse_cpp_log(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"raw_text": path.read_text(encoding="utf-8")}
    matrix: list[list[int]] = []
    in_matrix = False
    for line in info["raw_text"].splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("batch_predictions:"):
            info["batch_predictions"] = line.split(":", 1)[1].strip()
            continue
        if line.startswith("samples="):
            for token in line.split():
                key, value = token.split("=", 1)
                if key in {"samples", "correct"}:
                    info[key] = int(value)
                elif key == "accuracy":
                    info[key] = float(value)
            continue
        if line == "[CONFUSION_MATRIX]":
            in_matrix = True
            continue
        if in_matrix:
            matrix.append([int(part) for part in line.split()])
    if matrix:
        info["confusion_matrix"] = matrix
    return info


def format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def relative_href(base_file: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_file.resolve().parent).replace("\\", "/")


def class_color(value: float) -> str:
    clamped = max(0.0, min(1.0, value))
    red = int(245 - 90 * clamped)
    green = int(244 - 35 * clamped)
    blue = int(238 - 170 * clamped)
    return f"rgb({red}, {green}, {blue})"


def build_confusion_from_predictions(rows: list[PredictionRow]) -> list[list[int]]:
    matrix = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    for row in rows:
        matrix[row.target_idx][row.pred_idx] += 1
    return matrix


def load_run_metadata(run_dir: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "train_samples": 45000,
        "val_samples": 5000,
        "test_samples": 10000,
        "phase1_epochs": 10,
        "phase2_epochs": 20,
        "config": {
            "seed": 42,
            "num_classes": 10,
            "image_size": 224,
            "batch_size": 64,
            "num_workers": 4,
            "epochs_phase1": 10,
            "epochs_phase2": 20,
            "lr_phase1": 0.0001,
            "lr_phase2": 0.00005,
            "weight_decay": 0.0001,
            "model": {
                "backbone": "resnet18",
                "pretrained": True,
                "embed_dim": 512,
                "transformer_depth": 2,
                "num_heads": 8,
                "mlp_ratio": 4.0,
                "dropout": 0.1,
            },
        },
    }
    config_path = Path("configs/cifar10_hybrid_full.yaml")
    if yaml is not None and config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            metadata["config"] = loaded
            metadata["phase1_epochs"] = int(loaded.get("epochs_phase1", metadata["phase1_epochs"]))
            metadata["phase2_epochs"] = int(loaded.get("epochs_phase2", metadata["phase2_epochs"]))
    if torch is None:
        return metadata

    checkpoint_candidates = (
        run_dir / "best.pth",
        run_dir / "final_hybrid.pth",
        run_dir / "best_phase2.pth",
    )
    for checkpoint_path in checkpoint_candidates:
        if not checkpoint_path.exists():
            continue
        try:
            payload = torch.load(checkpoint_path, map_location="cpu")
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        cfg = payload.get("config")
        if not isinstance(cfg, dict):
            continue
        metadata["phase1_epochs"] = int(cfg.get("epochs_phase1", metadata["phase1_epochs"]))
        metadata["phase2_epochs"] = int(cfg.get("epochs_phase2", metadata["phase2_epochs"]))
        metadata["config"] = cfg
        break
    return metadata


def render_config_cards(cfg: dict[str, Any]) -> str:
    model_cfg = cfg.get("model", {})
    items = [
        ("seed", cfg.get("seed")),
        ("num_classes", cfg.get("num_classes")),
        ("image_size", cfg.get("image_size")),
        ("batch_size", cfg.get("batch_size")),
        ("num_workers", cfg.get("num_workers")),
        ("epochs_phase1", cfg.get("epochs_phase1")),
        ("epochs_phase2", cfg.get("epochs_phase2")),
        ("lr_phase1", cfg.get("lr_phase1")),
        ("lr_phase2", cfg.get("lr_phase2")),
        ("weight_decay", cfg.get("weight_decay")),
        ("backbone", model_cfg.get("backbone")),
        ("pretrained", model_cfg.get("pretrained")),
        ("embed_dim", model_cfg.get("embed_dim")),
        ("transformer_depth", model_cfg.get("transformer_depth")),
        ("num_heads", model_cfg.get("num_heads")),
        ("mlp_ratio", model_cfg.get("mlp_ratio")),
        ("dropout", model_cfg.get("dropout")),
    ]
    cards = []
    for label, value in items:
        cards.append(
            '<article class="config-card">'
            f'<span class="config-label">{html.escape(str(label))}</span>'
            f'<span class="config-value">{html.escape(str(value))}</span>'
            '</article>'
        )
    return "".join(cards)


def copy_architecture_image(output_path: Path) -> Path | None:
    source_path = Path(__file__).resolve().parents[1] / "src" / "hybrid_model.png"
    if not source_path.exists():
        return None
    asset_dir = output_path.resolve().parent / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    target_path = asset_dir / "hybrid_model.png"
    shutil.copy2(source_path, target_path)
    return target_path


def render_architecture_image(output_path: Path) -> str:
    image_path = copy_architecture_image(output_path)
    if image_path is None:
        return (
            '<div class="empty-state">'
            'Missing architecture image: <code>src/hybrid_model.png</code>'
            "</div>"
        )
    image_src = Path(os.path.relpath(image_path, start=output_path.resolve().parent))
    return (
        '<div class="architecture-image-wrap">'
        f'<img class="architecture-image" src="{image_src.as_posix()}" alt="Hybrid model architecture">'
        "</div>"
    )


def svg_bar_chart(counts: dict[str, int], title: str) -> str:
    width = 860
    height = 250
    margin = 38
    items = list(counts.items())
    max_value = max(counts.values()) if counts else 1
    step = (width - margin * 2) / max(len(items), 1)
    bars: list[str] = [
        f'<text x="{margin}" y="20" fill="#1c232b" font-size="16" font-weight="700">{html.escape(title)}</text>'
    ]
    for index, (label, value) in enumerate(items):
        x = margin + index * step + step * 0.1
        bar_width = step * 0.8
        bar_height = 0 if max_value == 0 else (height - 90) * (value / max_value)
        y = height - 36 - bar_height
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" '
            f'rx="8" fill="#2f6fed" opacity="0.9"></rect>'
        )
        bars.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{height - 14}" text-anchor="middle" '
            f'fill="#435160" font-size="12">{html.escape(label)}</text>'
        )
        bars.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{max(y - 8, 34):.1f}" text-anchor="middle" '
            f'fill="#1c232b" font-size="12">{value}</text>'
        )
    bars.append(
        f'<line x1="{margin}" y1="{height - 36}" x2="{width - margin}" y2="{height - 36}" '
        'stroke="#a8b3bf" stroke-width="1"></line>'
    )
    return f'<svg viewBox="0 0 {width} {height}" class="chart">{"".join(bars)}</svg>'


def render_confusion_matrix(matrix: list[list[int]]) -> str:
    row_maxes = [max(row) if row else 1 for row in matrix]
    header_cells = "".join(f"<th>{html.escape(name[:5])}</th>" for name in CLASS_NAMES)
    body_rows = []
    for row_index, row in enumerate(matrix):
        cells = []
        row_max = row_maxes[row_index] or 1
        for value in row:
            shade = class_color(value / row_max)
            text_color = "#ffffff" if value / row_max > 0.6 else "#1f252d"
            cells.append(
                f'<td style="background:{shade};color:{text_color}">{value}</td>'
            )
        body_rows.append(
            f"<tr><th>{html.escape(CLASS_NAMES[row_index][:5])}</th>{''.join(cells)}</tr>"
        )
    return (
        '<div class="matrix-wrap">'
        '<table class="matrix">'
        f"<thead><tr><th>gt\\pred</th>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def image_src_for_html(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    mime_type = mime_type or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def render_gallery(title: str, rows: list[PredictionRow], output_path: Path) -> str:
    cards: list[str] = []
    for row in rows:
        img_href = image_src_for_html(row.image_path.resolve())
        correctness = "correct" if row.correct else "wrong"
        cards.append(
            '<figure class="sample-card">'
            f'<img src="{html.escape(img_href)}" alt="{html.escape(row.target_name)}">'
            '<figcaption>'
            f'<div class="sample-head {correctness}">{html.escape(row.target_name)} -> {html.escape(row.pred_name)}</div>'
            f'<div>confidence: {row.confidence:.4f}</div>'
            f"<div>dataset_index: {row.dataset_index}</div>"
            f"<div>order: {row.order}</div>"
            "</figcaption>"
            "</figure>"
        )
    if not cards:
        cards.append('<div class="empty-state">No samples for this section.</div>')
    return (
        '<section class="panel">'
        f"<h2>{html.escape(title)}</h2>"
        f'<div class="gallery">{"".join(cards)}</div>'
        "</section>"
    )


def build_report(
    rows: list[PredictionRow],
    log_info: dict[str, Any],
    run_metadata: dict[str, Any],
    output_path: Path,
    top_k: int,
) -> str:
    total = len(rows)
    correct = sum(1 for row in rows if row.correct)
    accuracy = correct / total
    wrong_rows = [row for row in rows if not row.correct]
    correct_rows = [row for row in rows if row.correct]
    avg_conf = sum(row.confidence for row in rows) / total
    avg_conf_correct = sum(row.confidence for row in correct_rows) / max(len(correct_rows), 1)
    avg_conf_wrong = sum(row.confidence for row in wrong_rows) / max(len(wrong_rows), 1)
    confusion = build_confusion_from_predictions(rows)
    log_confusion = log_info.get("confusion_matrix")

    per_class: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        subset = [row for row in rows if row.target_idx == class_index]
        class_total = len(subset)
        class_correct = sum(1 for row in subset if row.correct)
        class_accuracy = class_correct / class_total if class_total else 0.0
        avg_class_conf = sum(row.confidence for row in subset) / class_total if class_total else 0.0
        common_wrong = Counter(row.pred_name for row in subset if not row.correct).most_common(1)
        per_class.append(
            {
                "name": class_name,
                "total": class_total,
                "correct": class_correct,
                "accuracy": class_accuracy,
                "avg_conf": avg_class_conf,
                "common_wrong": common_wrong[0][0] if common_wrong else "-",
                "common_wrong_count": common_wrong[0][1] if common_wrong else 0,
            }
        )

    confusion_pairs = Counter(
        (row.target_name, row.pred_name) for row in wrong_rows
    ).most_common(12)

    prediction_counts = Counter(row.pred_name for row in rows)
    target_counts = Counter(row.target_name for row in rows)

    hardest_errors = sorted(wrong_rows, key=lambda row: row.confidence, reverse=True)[:top_k]
    least_confident = sorted(rows, key=lambda row: row.confidence)[:top_k]
    strongest_correct = sorted(correct_rows, key=lambda row: row.confidence, reverse=True)[:top_k]

    log_accuracy = log_info.get("accuracy")
    accuracy_match = (
        log_accuracy is not None and abs(log_accuracy - accuracy) < 1e-9
    )
    log_note = (
        "matches predictions.tsv"
        if accuracy_match
        else "does not match predictions.tsv"
    )

    class_rows_html = "".join(
        "<tr>"
        f"<td>{html.escape(entry['name'])}</td>"
        f"<td>{entry['correct']} / {entry['total']}</td>"
        f"<td>{entry['accuracy']:.4f}</td>"
        f"<td>{entry['avg_conf']:.4f}</td>"
        f"<td>{html.escape(entry['common_wrong'])}</td>"
        f"<td>{entry['common_wrong_count']}</td>"
        "</tr>"
        for entry in per_class
    )

    confusion_pair_html = "".join(
        "<tr>"
        f"<td>{html.escape(target)}</td>"
        f"<td>{html.escape(pred)}</td>"
        f"<td>{count}</td>"
        "</tr>"
        for (target, pred), count in confusion_pairs
    )

    train_samples = int(run_metadata["train_samples"])
    val_samples = int(run_metadata["val_samples"])
    test_samples = int(run_metadata["test_samples"])
    phase1_epochs = int(run_metadata["phase1_epochs"])
    phase2_epochs = int(run_metadata["phase2_epochs"])
    total_epochs = phase1_epochs + phase2_epochs
    cfg = run_metadata["config"]
    architecture_image = render_architecture_image(output_path)
    config_cards = render_config_cards(cfg)

    html_report = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>C++ CIFAR-10 Prediction Report</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: rgba(255, 252, 247, 0.92);
      --ink: #1d232b;
      --muted: #5b6672;
      --line: #d9cdbf;
      --accent: #b45f2f;
      --accent-2: #2f6fed;
      --good: #167b46;
      --bad: #b22f2f;
      --shadow: 0 18px 40px rgba(47, 35, 21, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Noto Sans", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(180, 95, 47, 0.18), transparent 26%),
        radial-gradient(circle at top right, rgba(47, 111, 237, 0.16), transparent 22%),
        linear-gradient(180deg, #f7f1e8 0%, #f0ebe1 100%);
    }}
    .page {{
      width: min(1280px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0 56px;
    }}
    .hero {{
      padding: 28px;
      border: 1px solid rgba(217, 205, 191, 0.7);
      border-radius: 24px;
      background: linear-gradient(135deg, rgba(255, 252, 247, 0.95), rgba(249, 242, 232, 0.9));
      box-shadow: var(--shadow);
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: clamp(2rem, 4vw, 3.3rem);
      line-height: 1.02;
      letter-spacing: -0.04em;
    }}
    .hero p {{
      margin: 0;
      max-width: 900px;
      color: var(--muted);
      line-height: 1.6;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-top: 20px;
    }}
    .stat-card {{
      padding: 16px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.75);
      border: 1px solid rgba(217, 205, 191, 0.75);
    }}
    .stat-card .label {{
      display: block;
      color: var(--muted);
      font-size: 0.86rem;
      margin-bottom: 8px;
    }}
    .stat-card .value {{
      font-size: 1.8rem;
      font-weight: 700;
      letter-spacing: -0.04em;
    }}
    main {{
      display: grid;
      gap: 18px;
      margin-top: 20px;
    }}
    .panel {{
      padding: 22px;
      border-radius: 22px;
      background: var(--panel);
      border: 1px solid rgba(217, 205, 191, 0.82);
      box-shadow: var(--shadow);
    }}
    .panel h2 {{
      margin: 0 0 14px;
      font-size: 1.35rem;
      letter-spacing: -0.03em;
    }}
    .two-up {{
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 18px;
    }}
    .three-up {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 18px;
    }}
    .callout {{
      padding: 14px 16px;
      border-left: 4px solid var(--accent);
      background: rgba(180, 95, 47, 0.08);
      border-radius: 14px;
      color: #54311d;
      line-height: 1.6;
    }}
    .config-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
    }}
    .config-card {{
      padding: 14px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.82);
      border: 1px solid rgba(217, 205, 191, 0.82);
    }}
    .config-label {{
      display: block;
      color: var(--muted);
      font-size: 0.85rem;
      margin-bottom: 8px;
    }}
    .config-value {{
      display: block;
      font-size: 1.1rem;
      font-weight: 700;
      color: var(--ink);
      word-break: break-word;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      font-size: 0.95rem;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
    }}
    .matrix-wrap {{
      overflow-x: auto;
    }}
    .matrix th, .matrix td {{
      text-align: center;
      min-width: 56px;
      border: 1px solid rgba(217, 205, 191, 0.9);
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 14px;
    }}
    .sample-card {{
      margin: 0;
      overflow: hidden;
      border-radius: 16px;
      border: 1px solid rgba(217, 205, 191, 0.82);
      background: rgba(255, 255, 255, 0.86);
    }}
    .sample-card img {{
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      background: linear-gradient(180deg, #f9f5ef, #ece5d9);
      image-rendering: pixelated;
      padding: 10px;
    }}
    .sample-card figcaption {{
      padding: 12px;
      font-size: 0.9rem;
      color: var(--muted);
      line-height: 1.45;
    }}
    .sample-head {{
      margin-bottom: 6px;
      color: var(--ink);
      font-weight: 700;
    }}
    .sample-head.correct {{ color: var(--good); }}
    .sample-head.wrong {{ color: var(--bad); }}
    .empty-state {{
      padding: 20px;
      border: 1px dashed var(--line);
      border-radius: 16px;
      color: var(--muted);
    }}
    .architecture-image-wrap {{
      border-radius: 22px;
      overflow: hidden;
      border: 1px solid rgba(217, 205, 191, 0.82);
      background: rgba(255, 255, 255, 0.7);
    }}
    .architecture-image {{
      width: 100%;
      height: auto;
      display: block;
    }}
    @media (max-width: 960px) {{
      .two-up, .three-up {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>C++ CIFAR-10 Batch Inference Report</h1>
      <p>
        This report summarizes the full CIFAR-10 workflow with only the final essentials:
        how many samples were used, in what order the run proceeded, and what final accuracy
        the C++ inference stage achieved.
      </p>
      <div class="summary-grid">
        <article class="stat-card"><span class="label">Train samples</span><span class="value">{train_samples}</span></article>
        <article class="stat-card"><span class="label">Validation samples</span><span class="value">{val_samples}</span></article>
        <article class="stat-card"><span class="label">Test samples</span><span class="value">{test_samples}</span></article>
        <article class="stat-card"><span class="label">Total epochs</span><span class="value">{total_epochs}</span></article>
        <article class="stat-card"><span class="label">Correct</span><span class="value">{correct}</span></article>
        <article class="stat-card"><span class="label">Accuracy</span><span class="value">{accuracy:.4f}</span></article>
      </div>
    </section>
    <main>
      <section class="panel">
        <h2>Final Conclusion</h2>
        <div class="callout">
          전체 학습은 <strong>{train_samples}장</strong>으로 진행했고, 검증은 <strong>{val_samples}장</strong>,
          최종 C++ 테스트 평가는 <strong>{test_samples}장</strong>으로 수행했습니다.<br>
          순서는 <strong>1) phase1 {phase1_epochs} epoch</strong>에서 backbone을 고정한 학습,
          <strong>2) phase2 {phase2_epochs} epoch</strong>에서 전체 fine-tuning,
          <strong>3) TorchScript export</strong>, <strong>4) C++ batch inference</strong> 입니다.<br>
          최종 결과는 <strong>{correct} / {total}</strong> 정답, <strong>accuracy={accuracy:.4f}</strong> 입니다.
        </div>
      </section>

      <section class="panel">
        <h2>End-to-End Architecture</h2>
        {architecture_image}
      </section>

      <section class="panel">
        <h2>Configuration Visualization</h2>
        <div class="callout">
          학습, 토큰화, Transformer 깊이, head 수, dropout, learning rate까지 현재 실행 설정값을 한 눈에 보이도록 정리했습니다.
        </div>
        <div class="config-grid">{config_cards}</div>
      </section>

      <section class="panel">
        <h2>Prediction Distribution</h2>
        {svg_bar_chart(dict(target_counts), "Ground-truth sample counts by class")}
        {svg_bar_chart(dict(prediction_counts), "Predicted sample counts by class")}
      </section>

      <section class="panel">
        <h2>Confusion Matrix</h2>
        <div class="callout">
          Rows are ground-truth classes and columns are predicted classes. Darker cells mean
          more samples within that ground-truth row. The matrix below is recomputed from
          <code>predictions.tsv</code>.
        </div>
        {render_confusion_matrix(confusion)}
      </section>

      <section class="panel">
        <h2>Top Error Directions</h2>
        <table>
          <thead>
            <tr><th>Ground truth</th><th>Predicted</th><th>Count</th></tr>
          </thead>
          <tbody>{confusion_pair_html}</tbody>
        </table>
      </section>

      <section class="panel">
        <h2>Class-Level Performance</h2>
        <table>
          <thead>
            <tr>
              <th>Class</th>
              <th>Correct / Total</th>
              <th>Recall</th>
              <th>Avg confidence</th>
              <th>Most common wrong prediction</th>
              <th>Wrong count</th>
            </tr>
          </thead>
          <tbody>{class_rows_html}</tbody>
        </table>
      </section>

      {render_gallery("Most Confident Wrong Predictions", hardest_errors, output_path)}
      {render_gallery("Lowest Confidence Predictions", least_confident, output_path)}
      {render_gallery("Most Confident Correct Predictions", strongest_correct, output_path)}
    </main>
  </div>
</body>
</html>
"""

    if log_confusion and log_confusion != confusion:
        raise ValueError("Confusion matrix in cpp_infer.log does not match predictions.tsv")

    return html_report


def main() -> None:
    args = parse_args()
    rows = load_predictions(args.predictions)
    log_info = parse_cpp_log(args.log)
    run_metadata = load_run_metadata(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(rows, log_info, run_metadata, args.output, args.top_k)
    args.output.write_text(report, encoding="utf-8")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
