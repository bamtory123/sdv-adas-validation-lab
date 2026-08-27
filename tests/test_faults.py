from sdv_adas_validation_lab.contract import SensorFrame
from sdv_adas_validation_lab.faults import DelayQueue, FaultConfig


class Clock:
  def __init__(self) -> None:
    self.now = 1

  def __call__(self) -> int:
    return self.now


def frame(frame_id: int) -> SensorFrame:
  return SensorFrame(frame_id, frame_id + 1, b"rgb", 1, 1)


def test_zero_delay_uses_queue_and_publishes() -> None:
  clock = Clock()
  queue = DelayQueue(FaultConfig(), clock=clock)
  assert queue.submit(frame(0)).kind == "queued"
  assert [item.frame.frame_id for item in queue.publish_ready()] == [0]


def test_deadline_and_fifo_are_preserved() -> None:
  clock = Clock()
  queue = DelayQueue(FaultConfig(delay_ms=50), clock=clock)
  queue.submit(frame(0))
  queue.submit(frame(1))
  clock.now += 49_999_999
  assert queue.publish_ready() == []
  clock.now += 1
  assert [item.frame.frame_id for item in queue.publish_ready()] == [0, 1]


def test_configured_drop_and_overflow_are_explicit() -> None:
  clock = Clock()
  dropped = DelayQueue(FaultConfig(drop_every=2), clock=clock)
  assert dropped.submit(frame(0)).kind == "queued"
  assert dropped.submit(frame(1)).reason == "configured_drop"
  full = DelayQueue(FaultConfig(max_pending=1, delay_ms=1), clock=clock)
  full.submit(frame(0))
  assert full.submit(frame(1)).reason == "queue_overflow"
