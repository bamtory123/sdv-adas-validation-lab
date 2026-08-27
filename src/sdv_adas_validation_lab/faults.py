"""Non-blocking, deterministic transport delay and frame-drop injector."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from time import monotonic_ns

from .contract import SensorFrame


@dataclass(frozen=True, slots=True)
class FaultConfig:
  delay_ms: int = 0
  drop_every: int = 0
  max_pending: int = 256

  def validate(self) -> None:
    if self.delay_ms < 0:
      raise ValueError("delay_ms must be non-negative")
    if self.drop_every < 0:
      raise ValueError("drop_every must be non-negative")
    if self.max_pending <= 0:
      raise ValueError("max_pending must be positive")


@dataclass(frozen=True, slots=True)
class FrameEvent:
  frame_id: int
  kind: str
  timestamp_ns: int
  target_delay_ns: int
  actual_delay_ns: int | None = None
  queue_depth: int = 0
  reason: str | None = None


@dataclass(frozen=True, slots=True)
class PublishedFrame:
  frame: SensorFrame
  event: FrameEvent


class DelayQueue:
  """Producer never sleeps; a separate publisher polls ready frames by deadline."""

  def __init__(self, config: FaultConfig, *, clock=monotonic_ns) -> None:
    config.validate()
    self.config = config
    self._clock = clock
    self._pending: list[tuple[int, int, SensorFrame, int]] = []
    self._submission_count = 0

  def submit(self, frame: SensorFrame) -> FrameEvent:
    """Accept or drop one frame immediately. This method never waits for delay."""
    now_ns = self._clock()
    self._submission_count += 1
    target_delay_ns = self.config.delay_ms * 1_000_000
    if self.config.drop_every and self._submission_count % self.config.drop_every == 0:
      return FrameEvent(frame.frame_id, "dropped", now_ns, target_delay_ns, queue_depth=len(self._pending), reason="configured_drop")
    if len(self._pending) >= self.config.max_pending:
      return FrameEvent(frame.frame_id, "invalid", now_ns, target_delay_ns, queue_depth=len(self._pending), reason="queue_overflow")
    deadline_ns = now_ns + target_delay_ns
    heapq.heappush(self._pending, (deadline_ns, frame.frame_id, frame, now_ns))
    return FrameEvent(frame.frame_id, "queued", now_ns, target_delay_ns, queue_depth=len(self._pending))

  def publish_ready(self) -> list[PublishedFrame]:
    now_ns = self._clock()
    published: list[PublishedFrame] = []
    while self._pending and self._pending[0][0] <= now_ns:
      deadline_ns, _, frame, entered_ns = heapq.heappop(self._pending)
      published.append(
        PublishedFrame(
          frame,
          FrameEvent(
            frame.frame_id,
            "published",
            now_ns,
            deadline_ns - entered_ns,
            actual_delay_ns=now_ns - entered_ns,
            queue_depth=len(self._pending),
          ),
        )
      )
    return published

  @property
  def pending_count(self) -> int:
    return len(self._pending)

  def shutdown(self) -> list[FrameEvent]:
    now_ns = self._clock()
    events = [
      FrameEvent(frame.frame_id, "invalid", now_ns, deadline_ns - entered_ns, queue_depth=0, reason="shutdown")
      for deadline_ns, _, frame, entered_ns in self._pending
    ]
    self._pending.clear()
    return events
