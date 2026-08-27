import pytest

from sdv_adas_validation_lab import FrameStatus, SensorFrame
from sdv_adas_validation_lab.integrity import validate_stream


def frame(frame_id: int, timestamp: int) -> SensorFrame:
  return SensorFrame(frame_id, timestamp, b"rgb", 1, 1)


def test_valid_frame_and_stream() -> None:
  validate_stream([frame(0, 1), frame(1, 2)])


@pytest.mark.parametrize(
  "bad",
  [
    SensorFrame(-1, 1, b"rgb", 1, 1),
    SensorFrame(0, 0, b"rgb", 1, 1),
    SensorFrame(0, 1, b"", 1, 1),
    SensorFrame(0, 1, b"rgb", 0, 1),
  ],
)
def test_invalid_frame_is_rejected(bad: SensorFrame) -> None:
  with pytest.raises(ValueError):
    bad.validate()


def test_dropped_frame_can_have_no_image() -> None:
  SensorFrame(0, 1, b"", 1, 1, status=FrameStatus.DROPPED).validate()


def test_non_monotonic_stream_is_rejected() -> None:
  with pytest.raises(ValueError, match="increase strictly"):
    validate_stream([frame(0, 2), frame(1, 1)])
