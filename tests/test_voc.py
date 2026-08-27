import json

import numpy as np
from PIL import Image

from sdv_adas_validation_lab.voc import VOC_CLASS_COUNT, VOC_IGNORE_LABEL, build_labeled_fixture


def test_build_labeled_fixture_converts_voc_pair(tmp_path) -> None:
  root = tmp_path / "VOC2012"
  (root / "JPEGImages").mkdir(parents=True)
  (root / "SegmentationClass").mkdir()
  Image.fromarray(np.array([[[1, 2, 3]]], dtype=np.uint8)).save(root / "JPEGImages" / "sample.jpg")
  Image.fromarray(np.array([[7]], dtype=np.uint8)).save(root / "SegmentationClass" / "sample.png")

  manifest = build_labeled_fixture(root, ["sample"], tmp_path / "fixture")

  record = json.loads(manifest.read_text(encoding="utf-8"))
  assert record["source_image_id"] == "sample"
  assert (tmp_path / "fixture" / record["image_path"]).read_bytes()
  assert VOC_CLASS_COUNT == 21
  assert VOC_IGNORE_LABEL == 255
