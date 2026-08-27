"""Repeatable replay-and-fault harness; inference will plug in after publication."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from .faults import DelayQueue, FaultConfig, FrameEvent
from .preflight import collect
from .replay import ReplaySource


def _write_events(path: Path, events: list[FrameEvent]) -> None:
  with path.open("w", encoding="utf-8") as output:
    for event in events:
      output.write(json.dumps(asdict(event), sort_keys=True) + "\n")


def run_once(source: ReplaySource, fault: FaultConfig, output_dir: Path) -> dict[str, object]:
  output_dir.mkdir(parents=True, exist_ok=False)
  queue = DelayQueue(fault)
  events: list[FrameEvent] = []
  first_capture_ns = next(iter(source)).capture_monotonic_ns
  start_ns = time.monotonic_ns()
  for frame in source:
    due_ns = start_ns + frame.capture_monotonic_ns - first_capture_ns
    while time.monotonic_ns() < due_ns:
      events.extend(item.event for item in queue.publish_ready())
      time.sleep(0.001)
    events.append(queue.submit(frame))
    events.extend(item.event for item in queue.publish_ready())
  while queue.pending_count:
    events.extend(item.event for item in queue.publish_ready())
    if queue.pending_count:
      time.sleep(0.001)

  invalid = [event for event in events if event.kind == "invalid"]
  published = [event for event in events if event.kind == "published"]
  dropped = [event for event in events if event.kind == "dropped"]
  with (output_dir / "frames.csv").open("w", newline="", encoding="utf-8") as output:
    writer = csv.DictWriter(output, fieldnames=list(asdict(events[0]).keys()))
    writer.writeheader()
    writer.writerows(asdict(event) for event in events)
  _write_events(output_dir / "events.jsonl", events)
  summary = {
    "validity": "invalid" if invalid else "valid",
    "outcome": "not_evaluated" if invalid else "pass",
    "captured_frames": len(source),
    "published_frames": len(published),
    "dropped_frames": len(dropped),
    "invalid_events": len(invalid),
  }
  (output_dir / "manifest.json").write_text(
    json.dumps({"preflight": collect(), "fault": asdict(fault), "source_frames": len(source)}, indent=2) + "\n",
    encoding="utf-8",
  )
  (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
  return summary


def run_repeated(source: ReplaySource, fault: FaultConfig, output_root: Path, repeats: int) -> list[dict[str, object]]:
  if repeats <= 0:
    raise ValueError("repeats must be positive")
  return [run_once(source, fault, output_root / f"run-{index:03d}-{uuid4().hex[:8]}") for index in range(1, repeats + 1)]


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--replay", type=Path, help="JSONL replay manifest")
  parser.add_argument("--synthetic-frames", type=int, default=0)
  parser.add_argument("--delay-ms", type=int, default=0)
  parser.add_argument("--drop-every", type=int, default=0)
  parser.add_argument("--repeats", type=int, default=1)
  parser.add_argument("--output", type=Path, default=Path("outputs/replay-fault"))
  args = parser.parse_args()
  source = ReplaySource.from_jsonl(args.replay) if args.replay else ReplaySource.synthetic(args.synthetic_frames or 20)
  summaries = run_repeated(source, FaultConfig(args.delay_ms, args.drop_every), args.output, args.repeats)
  print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
  main()
