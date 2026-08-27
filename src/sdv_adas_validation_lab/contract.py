"""Versioned, immutable camera-frame contract used between all pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


CONTRACT_VERSION = "sensor-frame/v1"


class FrameStatus(StrEnum):
  OK = "ok"
  DROPPED = "dropped"
  INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class SensorFrame:
  """One camera frame; timestamps are monotonic nanoseconds."""

  frame_id: int
  capture_monotonic_ns: int
  image: bytes
  width: int
  height: int
  pixel_format: str = "rgb8"
  status: FrameStatus = FrameStatus.OK

  def validate(self) -> None:
    if self.frame_id < 0:
      raise ValueError("frame_id must be non-negative")
    if self.capture_monotonic_ns <= 0:
      raise ValueError("capture_monotonic_ns must be positive")
    if self.width <= 0 or self.height <= 0:
      raise ValueError("image dimensions must be positive")
    if self.status is FrameStatus.OK and not self.image:
      raise ValueError("an OK frame must contain image bytes")
