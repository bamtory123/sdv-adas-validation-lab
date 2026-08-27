"""Deterministic file-backed camera replay input."""

from __future__ import annotations

import json
from pathlib import Path

from .contract import FrameStatus, SensorFrame
from .integrity import validate_stream


class ReplaySource:
  """A validated sequence of camera frames loaded before execution begins."""

  def __init__(self, frames: list[SensorFrame]) -> None:
    validate_stream(frames)
    self._frames = tuple(frames)

  def __iter__(self):
    return iter(self._frames)

  def __len__(self) -> int:
    return len(self._frames)

  @classmethod
  def from_jsonl(cls, manifest_path: Path) -> "ReplaySource":
    frames: list[SensorFrame] = []
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
      if not line.strip():
        continue
      item = json.loads(line)
      image_path = manifest_path.parent / item["image_path"]
      try:
        image = image_path.read_bytes()
      except FileNotFoundError as exc:
        raise ValueError(f"missing replay image at line {line_number}: {image_path}") from exc
      frames.append(
        SensorFrame(
          frame_id=item["frame_id"],
          capture_monotonic_ns=item["capture_monotonic_ns"],
          image=image,
          width=item["width"],
          height=item["height"],
          pixel_format=item.get("pixel_format", "rgb8"),
          status=FrameStatus(item.get("status", "ok")),
        )
      )
    if not frames:
      raise ValueError("replay manifest contains no frames")
    return cls(frames)

  @classmethod
  def synthetic(cls, frame_count: int, *, width: int = 2, height: int = 2, period_ns: int = 50_000_000) -> "ReplaySource":
    if frame_count <= 0:
      raise ValueError("frame_count must be positive")
    size = width * height * 3
    return cls(
      [
        SensorFrame(i, 1_000_000_000 + i * period_ns, bytes([i % 256]) * size, width, height)
        for i in range(frame_count)
      ]
    )
