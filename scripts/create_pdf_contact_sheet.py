"""Create a compact, labelled contact sheet from rendered PDF page PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=320)
    args = parser.parse_args()

    pages = sorted(args.input_dir.glob("page-*.png"))
    if not pages:
        raise SystemExit(f"No page PNGs found in {args.input_dir}")

    thumbs: list[Image.Image] = []
    for page in pages:
        image = Image.open(page).convert("RGB")
        height = round(image.height * args.thumb_width / image.width)
        image.thumbnail((args.thumb_width, height), Image.Resampling.LANCZOS)
        thumbs.append(image.copy())

    label_height = 28
    gap = 14
    cell_width = args.thumb_width + gap
    cell_height = max(image.height for image in thumbs) + label_height + gap
    rows = (len(thumbs) + args.columns - 1) // args.columns
    sheet = Image.new(
        "RGB",
        (args.columns * cell_width + gap, rows * cell_height + gap),
        "#d9dde3",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)

    for index, image in enumerate(thumbs):
        row, column = divmod(index, args.columns)
        x = gap + column * cell_width
        y = gap + row * cell_height
        sheet.paste(image, (x, y + label_height))
        draw.text((x + 4, y + 3), f"Page {index + 1}", fill="#111827", font=font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=92)


if __name__ == "__main__":
    main()
