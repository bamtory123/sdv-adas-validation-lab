"""Prepare a local RGB replay fixture without committing raw source images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def prepare_fcn(images: list[Path], output_dir: Path, *, period_ns: int = 50_000_000) -> Path:
  output_dir.mkdir(parents=True, exist_ok=True)
  manifest = output_dir / "frames.jsonl"
  with manifest.open("w", encoding="utf-8") as output:
    for frame_id, image_path in enumerate(images):
      image = Image.open(image_path).convert("RGB")
      rgb_path = output_dir / f"frame-{frame_id:03d}.rgb"
      rgb_path.write_bytes(image.tobytes())
      output.write(
        json.dumps(
          {
            "frame_id": frame_id,
            "capture_monotonic_ns": 1_000_000_000 + frame_id * period_ns,
            "width": image.width,
            "height": image.height,
            "image_path": rgb_path.name,
          }
        )
        + "\n"
      )
  return manifest


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--image", action="append", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  print(prepare_fcn(args.image, args.output))


if __name__ == "__main__":
  main()
