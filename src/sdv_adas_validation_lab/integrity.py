"""Small, explicit validator for a replayed camera stream."""

from __future__ import annotations

from collections.abc import Iterable

from .contract import SensorFrame


def validate_stream(frames: Iterable[SensorFrame]) -> None:
  previous_id = -1
  previous_capture_ns = 0
  for frame in frames:
    frame.validate()
    if frame.frame_id <= previous_id:
      raise ValueError("frame_id must increase strictly")
    if frame.capture_monotonic_ns <= previous_capture_ns:
      raise ValueError("capture timestamps must increase strictly")
    previous_id = frame.frame_id
    previous_capture_ns = frame.capture_monotonic_ns
