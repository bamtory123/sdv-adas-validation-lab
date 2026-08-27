from PIL import Image

from sdv_adas_validation_lab.fixture import prepare_fcn
from sdv_adas_validation_lab.replay import ReplaySource


def test_fixture_writes_jsonl_and_rgb_frames(tmp_path) -> None:
  image = tmp_path / "input.png"
  Image.new("RGB", (2, 3), (1, 2, 3)).save(image)
  manifest = prepare_fcn([image], tmp_path / "fixture")
  frame = next(iter(ReplaySource.from_jsonl(manifest)))
  assert (frame.width, frame.height, len(frame.image)) == (2, 3, 18)
