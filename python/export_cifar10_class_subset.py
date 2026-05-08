"""고정 개수의 CIFAR-10 클래스 서브셋을 이미지 파일로 내보낸다."""

from __future__ import annotations

import argparse
from pathlib import Path

from torchvision import datasets


CIFAR10_CLASSES = (
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


def parse_args() -> argparse.Namespace:
    """명령행 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="Export CIFAR-10 samples for one class.")
    parser.add_argument("--class-name", type=str, default="dog")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=("test", "train"))
    return parser.parse_args()


def main() -> None:
    """요청된 CIFAR-10 클래스 서브셋을 내보낸다."""
    args = parse_args()
    if args.class_name not in CIFAR10_CLASSES:
        raise ValueError(
            f"Unsupported class '{args.class_name}'. Choose one of: {', '.join(CIFAR10_CLASSES)}"
        )

    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data" / "cifar_10"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.CIFAR10(
        root=str(data_root),
        train=(args.split == "train"),
        download=True,
    )
    class_idx = CIFAR10_CLASSES.index(args.class_name)
    indices = [idx for idx, target in enumerate(dataset.targets) if target == class_idx][: args.count]
    if not indices:
        raise RuntimeError(f"No samples found for class '{args.class_name}'.")

    manifest_path = output_dir / "manifest.tsv"
    lines = ["order\tdataset_index\tclass_name\tfile_name"]
    for order, dataset_index in enumerate(indices, start=1):
        image, _ = dataset[dataset_index]
        file_name = f"{order:02d}_idx{dataset_index:04d}_{args.class_name}.png"
        image.save(output_dir / file_name)
        lines.append(f"{order}\t{dataset_index}\t{args.class_name}\t{file_name}")

    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved={len(indices)}")
    print(f"output_dir={output_dir}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
