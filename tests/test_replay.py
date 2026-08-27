import json

import pytest

from sdv_adas_validation_lab.replay import ReplaySource


def test_synthetic_replay_is_repeatable() -> None:
  source = ReplaySource.synthetic(3)
  assert [frame.frame_id for frame in source] == [0, 1, 2]
  assert len(source) == 3
  assert source.content_hash == ReplaySource.synthetic(3).content_hash


def test_jsonl_replay_loads_image_bytes(tmp_path) -> None:
  (tmp_path / "frame.rgb").write_bytes(b"rgb")
  (tmp_path / "frames.jsonl").write_text(
    json.dumps({"frame_id": 0, "capture_monotonic_ns": 1, "width": 1, "height": 1, "image_path": "frame.rgb"}) + "\n"
  )
  assert next(iter(ReplaySource.from_jsonl(tmp_path / "frames.jsonl"))).image == b"rgb"


def test_empty_replay_is_rejected(tmp_path) -> None:
  path = tmp_path / "frames.jsonl"
  path.write_text("")
  with pytest.raises(ValueError, match="no frames"):
    ReplaySource.from_jsonl(path)
