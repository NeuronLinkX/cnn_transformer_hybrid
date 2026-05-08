"""이미지 디렉터리로부터 간단한 콘택트 시트를 만든다."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    """명령행 인자를 파싱한다."""
    parser = argparse.ArgumentParser(description="Create a contact sheet from images.")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--pattern", type=str, default="*.jpg")
    return parser.parse_args()


def main() -> None:
    """콘택트 시트를 생성한다."""
    args = parse_args()
    input_dir = Path(args.input_dir)
    image_paths = sorted(input_dir.glob(args.pattern))
    if not image_paths:
        raise RuntimeError(f"No images found in {input_dir} matching {args.pattern}.")

    first = Image.open(image_paths[0]).convert("RGB")
    tile_width, tile_height = first.size
    rows = math.ceil(len(image_paths) / args.columns)
    sheet = Image.new("RGB", (tile_width * args.columns, tile_height * rows), color=(24, 24, 24))

    for idx, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        x = (idx % args.columns) * tile_width
        y = (idx // args.columns) * tile_height
        sheet.paste(image, (x, y))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
