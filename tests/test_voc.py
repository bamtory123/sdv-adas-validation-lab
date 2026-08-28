import json

import numpy as np
from PIL import Image

import pytest

from sdv_adas_validation_lab.voc import VOC_CLASS_COUNT, VOC_IGNORE_LABEL, build_labeled_fixture, verify_trainval_archive


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


def test_archive_verification_rejects_unknown_content(tmp_path) -> None:
  archive = tmp_path / "VOCtrainval_11-May-2012.tar"
  archive.write_bytes(b"not a VOC archive")
  with pytest.raises(ValueError, match="MD5 mismatch"):
    verify_trainval_archive(archive)
