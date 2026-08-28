"""External PASCAL VOC 2012 segmentation fixture builder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from PIL import Image


VOC_CLASS_COUNT = 21
VOC_IGNORE_LABEL = 255
VOC2012_TRAINVAL_MD5 = "6cd6e144f989b92b3379bac3b3de84fd"


def verify_trainval_archive(path: Path) -> None:
  """Reject a truncated or wrong VOC 2012 train/validation archive."""
  digest = hashlib.md5(path.read_bytes()).hexdigest()
  if digest != VOC2012_TRAINVAL_MD5:
    raise ValueError(f"VOC 2012 trainval MD5 mismatch: {digest}")


def select_split_image_ids(
  voc_root: Path, split: str, count: int, *, seed: int, exclude: set[str] | None = None
) -> list[str]:
  """Select a deterministic, non-overlapping subset from a VOC segmentation split."""
  if count <= 0:
    raise ValueError("count must be positive")
  ids = (voc_root / "ImageSets" / "Segmentation" / f"{split}.txt").read_text(encoding="utf-8").splitlines()
  eligible = [image_id for image_id in ids if image_id not in (exclude or set())]
  if count > len(eligible):
    raise ValueError(f"requested {count} images but only {len(eligible)} are eligible")
  generator = random.Random(seed)
  generator.shuffle(eligible)
  return eligible[:count]


def build_labeled_fixture(
  voc_root: Path, image_ids: list[str], output_dir: Path, *, period_ns: int = 50_000_000, selection: dict[str, object] | None = None
) -> Path:
  """Convert selected VOC image/label pairs into the repository replay contract.

  `voc_root` is the external `VOCdevkit/VOC2012` directory. The fixture contains
  no model artifacts and must remain outside the repository.
  """
  if not image_ids:
    raise ValueError("at least one VOC image id is required")
  output_dir.mkdir(parents=True, exist_ok=False)
  records: list[dict[str, object]] = []
  for index, image_id in enumerate(image_ids):
    image_path = voc_root / "JPEGImages" / f"{image_id}.jpg"
    label_path = voc_root / "SegmentationClass" / f"{image_id}.png"
    if not image_path.is_file() or not label_path.is_file():
      raise ValueError(f"missing VOC image or class label for {image_id}")
    with Image.open(image_path) as image:
      rgb = image.convert("RGB")
      width, height = rgb.size
      raw_name = f"{image_id}.rgb"
      (output_dir / raw_name).write_bytes(rgb.tobytes())
    label_name = f"{image_id}.png"
    (output_dir / label_name).write_bytes(label_path.read_bytes())
    records.append(
      {
        "frame_id": index,
        "capture_monotonic_ns": 1_000_000_000 + index * period_ns,
        "image_path": raw_name,
        "label_path": label_name,
        "width": width,
        "height": height,
        "pixel_format": "rgb8",
        "source_image_id": image_id,
      }
    )
  manifest_path = output_dir / "frames.jsonl"
  manifest_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
  (output_dir / "source.json").write_text(
    json.dumps(
      {"dataset": "PASCAL VOC 2012", "class_count": VOC_CLASS_COUNT, "ignore_label": VOC_IGNORE_LABEL, "image_ids": image_ids, "selection": selection},
      indent=2,
    )
    + "\n",
    encoding="utf-8",
  )
  return manifest_path


def main() -> None:
  parser = argparse.ArgumentParser(description="Build a labeled replay from an external VOC 2012 directory.")
  parser.add_argument("--voc-root", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--image-id", action="append")
  parser.add_argument("--split", help="VOC segmentation split used with --sample-count")
  parser.add_argument("--sample-count", type=int)
  parser.add_argument("--seed", type=int, default=20260828)
  parser.add_argument("--exclude-image-id", action="append", default=[])
  parser.add_argument("--exclude-source", action="append", type=Path, default=[])
  args = parser.parse_args()
  if args.image_id and (args.split or args.sample_count):
    parser.error("--image-id cannot be combined with split sampling")
  if args.image_id:
    image_ids = args.image_id
    selection = {"method": "explicit_ids"}
  elif args.split and args.sample_count:
    excluded = set(args.exclude_image_id)
    for source_path in args.exclude_source:
      excluded.update(json.loads(source_path.read_text(encoding="utf-8"))["image_ids"])
    image_ids = select_split_image_ids(args.voc_root, args.split, args.sample_count, seed=args.seed, exclude=excluded)
    selection = {
      "method": "seeded_split",
      "split": args.split,
      "count": args.sample_count,
      "seed": args.seed,
      "exclude_image_ids": sorted(excluded),
      "exclude_sources": [str(path) for path in args.exclude_source],
    }
  else:
    parser.error("provide --image-id or both --split and --sample-count")
  print(build_labeled_fixture(args.voc_root, image_ids, args.output, selection=selection))


if __name__ == "__main__":
  main()
