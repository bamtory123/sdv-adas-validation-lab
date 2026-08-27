"""External PASCAL VOC 2012 segmentation fixture builder."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


VOC_CLASS_COUNT = 21
VOC_IGNORE_LABEL = 255


def build_labeled_fixture(voc_root: Path, image_ids: list[str], output_dir: Path, *, period_ns: int = 50_000_000) -> Path:
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
    json.dumps({"dataset": "PASCAL VOC 2012", "class_count": VOC_CLASS_COUNT, "ignore_label": VOC_IGNORE_LABEL, "image_ids": image_ids}, indent=2) + "\n",
    encoding="utf-8",
  )
  return manifest_path
