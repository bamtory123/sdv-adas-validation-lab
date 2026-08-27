"""Repeatable replay-and-fault harness; inference will plug in after publication."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from .contract import SensorFrame
from .faults import DelayQueue, FaultConfig, FrameEvent
from .preflight import collect
from .replay import ReplaySource
from .runtime import OnnxReferenceRuntime, Prediction
from .tensorrt_runtime import TensorRtRuntime


class RuntimeAdapter(Protocol):
  def infer(self, frame: SensorFrame) -> Prediction: ...


def _write_events(path: Path, events: list[FrameEvent]) -> None:
  with path.open("w", encoding="utf-8") as output:
    for event in events:
      output.write(json.dumps(asdict(event), sort_keys=True) + "\n")


def _timing_stats(values: list[int]) -> dict[str, float | None]:
  if not values:
    return {"median_ms": None, "p95_ms": None, "max_ms": None}
  ordered = sorted(values)
  p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
  return {
    "median_ms": statistics.median(ordered) / 1_000_000,
    "p95_ms": ordered[p95_index] / 1_000_000,
    "max_ms": ordered[-1] / 1_000_000,
  }


def _git_state() -> dict[str, object]:
  try:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.strip())
    return {"commit": commit, "dirty": dirty}
  except (OSError, subprocess.CalledProcessError):
    return {"commit": None, "dirty": None}


def _write_predictions(path: Path, predictions: list[Prediction]) -> None:
  with path.open("w", newline="", encoding="utf-8") as output:
    writer = csv.DictWriter(
      output, fieldnames=["frame_id", "started_ns", "finished_ns", "latency_ns", "output_shapes", "output_hashes", "primary_label_hash"]
    )
    writer.writeheader()
    writer.writerows(
      {
        "frame_id": prediction.frame_id,
        "started_ns": prediction.started_ns,
        "finished_ns": prediction.finished_ns,
        "latency_ns": prediction.latency_ns,
        "output_shapes": json.dumps(prediction.output_shapes),
        "output_hashes": json.dumps(prediction.output_hashes),
        "primary_label_hash": prediction.primary_label_hash,
      }
      for prediction in predictions
    )


def run_once(
  source: ReplaySource, fault: FaultConfig, output_dir: Path, runtime: RuntimeAdapter | None = None
) -> dict[str, object]:
  output_dir.mkdir(parents=True, exist_ok=False)
  queue = DelayQueue(fault)
  events: list[FrameEvent] = []
  predictions: list[Prediction] = []
  first_capture_ns = next(iter(source)).capture_monotonic_ns
  start_ns = time.monotonic_ns()
  for frame in source:
    due_ns = start_ns + frame.capture_monotonic_ns - first_capture_ns
    while time.monotonic_ns() < due_ns:
      ready = queue.publish_ready()
      events.extend(item.event for item in ready)
      if runtime:
        predictions.extend(runtime.infer(item.frame) for item in ready)
      time.sleep(0.001)
    events.append(queue.submit(frame))
    ready = queue.publish_ready()
    events.extend(item.event for item in ready)
    if runtime:
      predictions.extend(runtime.infer(item.frame) for item in ready)
  while queue.pending_count:
    ready = queue.publish_ready()
    events.extend(item.event for item in ready)
    if runtime:
      predictions.extend(runtime.infer(item.frame) for item in ready)
    if queue.pending_count:
      time.sleep(0.001)

  invalid = [event for event in events if event.kind == "invalid"]
  published = [event for event in events if event.kind == "published"]
  dropped = [event for event in events if event.kind == "dropped"]
  expected_frames = len(source) - len(dropped)
  unexpected_missing = max(0, expected_frames - len(published) - len(invalid))
  with (output_dir / "frames.csv").open("w", newline="", encoding="utf-8") as output:
    writer = csv.DictWriter(output, fieldnames=list(asdict(events[0]).keys()))
    writer.writeheader()
    writer.writerows(asdict(event) for event in events)
  _write_events(output_dir / "events.jsonl", events)
  if runtime:
    _write_predictions(output_dir / "predictions.csv", predictions)
  summary = {
    "validity": "invalid" if invalid else "valid",
    "outcome": "not_evaluated" if invalid else "fail" if unexpected_missing else "pass",
    "captured_frames": len(source),
    "published_frames": len(published),
    "dropped_frames": len(dropped),
    "expected_dropped_frames": len(dropped),
    "invalid_events": len(invalid),
    "unexpected_missing_frames": unexpected_missing,
    "inference_frames": len(predictions),
    "coverage": len(published) / expected_frames if expected_frames else 1.0,
    "transport_delay": _timing_stats([event.actual_delay_ns for event in published if event.actual_delay_ns is not None]),
    "inference_latency": _timing_stats([prediction.latency_ns for prediction in predictions]),
  }
  (output_dir / "manifest.json").write_text(
    json.dumps(
      {"preflight": collect(), "git": _git_state(), "fault": asdict(fault), "source_frames": len(source), "source_hash": source.content_hash},
      indent=2,
    )
    + "\n",
    encoding="utf-8",
  )
  (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
  return summary


def run_repeated(
  source: ReplaySource, fault: FaultConfig, output_root: Path, repeats: int, runtime: RuntimeAdapter | None = None
) -> list[dict[str, object]]:
  if repeats <= 0:
    raise ValueError("repeats must be positive")
  return [run_once(source, fault, output_root / f"run-{index:03d}-{uuid4().hex[:8]}", runtime) for index in range(1, repeats + 1)]


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--replay", type=Path, help="JSONL replay manifest")
  parser.add_argument("--synthetic-frames", type=int, default=0)
  parser.add_argument("--delay-ms", type=int, default=0)
  parser.add_argument("--drop-every", type=int, default=0)
  parser.add_argument("--max-pending", type=int, default=256)
  parser.add_argument("--repeats", type=int, default=1)
  parser.add_argument("--onnx-model", type=Path)
  parser.add_argument("--tensorrt-engine", type=Path)
  parser.add_argument("--provider", default="CPUExecutionProvider")
  parser.add_argument("--profile", choices=["unit", "fcn_resnet50_voc"], default="unit")
  parser.add_argument("--output", type=Path, default=Path("outputs/replay-fault"))
  args = parser.parse_args()
  source = ReplaySource.from_jsonl(args.replay) if args.replay else ReplaySource.synthetic(args.synthetic_frames or 20)
  if args.onnx_model and args.tensorrt_engine:
    parser.error("select only one runtime model")
  runtime = (
    OnnxReferenceRuntime(args.onnx_model, provider=args.provider, profile=args.profile)
    if args.onnx_model
    else TensorRtRuntime(args.tensorrt_engine, profile=args.profile)
    if args.tensorrt_engine
    else None
  )
  summaries = run_repeated(source, FaultConfig(args.delay_ms, args.drop_every, args.max_pending), args.output, args.repeats, runtime)
  print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
  main()
