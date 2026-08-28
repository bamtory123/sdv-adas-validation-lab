"""Batch runtime/delay experiment orchestration with an aggregate manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from uuid import uuid4

from .faults import FaultConfig
from .harness import run_once
from .replay import ReplaySource
from .tensorrt_runtime import TensorRtRuntime


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def run_delay_experiment(
  source: ReplaySource,
  engine_path: Path,
  output_dir: Path,
  *,
  delays_ms: list[int],
  repeats: int,
  warmups: int,
  profile: str,
  order_seed: int = 20260828,
) -> dict[str, object]:
  output_dir.mkdir(parents=True, exist_ok=False)
  if repeats <= 0 or warmups < 0:
    raise ValueError("repeats must be positive and warmups non-negative")
  runtime = TensorRtRuntime(engine_path, profile=profile)
  conditions = {delay_ms: output_dir / f"delay-{delay_ms:03d}ms" for delay_ms in delays_ms}
  for delay_ms in delays_ms:
    for warmup in range(1, warmups + 1):
      run_once(source, FaultConfig(delay_ms=delay_ms), conditions[delay_ms] / f"warmup-{warmup:03d}-{uuid4().hex[:8]}", runtime)
  generator = random.Random(order_seed)
  block_orders: list[list[int]] = []
  for block in range(1, repeats + 1):
    order = list(delays_ms)
    generator.shuffle(order)
    block_orders.append(order)
    for delay_ms in order:
      run_once(source, FaultConfig(delay_ms=delay_ms), conditions[delay_ms] / f"run-{block:03d}-{uuid4().hex[:8]}", runtime)
  manifest = {
    "kind": "delay-experiment/v1",
    "source_hash": source.content_hash,
    "engine_sha256": _sha256(engine_path),
    "profile": profile,
    "repeats": repeats,
    "warmups": warmups,
    "order_seed": order_seed,
    "block_orders_ms": block_orders,
    "conditions": [{"delay_ms": delay_ms, "measured_runs": repeats, "directory": conditions[delay_ms].name} for delay_ms in delays_ms],
  }
  (output_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  return manifest


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--replay", required=True, type=Path)
  parser.add_argument("--tensorrt-engine", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--delays-ms", type=int, nargs="+", default=[0, 50, 100, 150])
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--warmups", type=int, default=1)
  parser.add_argument("--order-seed", type=int, default=20260828)
  parser.add_argument("--profile", default="fcn_resnet50_voc")
  args = parser.parse_args()
  print(
    json.dumps(
      run_delay_experiment(
        ReplaySource.from_jsonl(args.replay),
        args.tensorrt_engine,
        args.output,
        delays_ms=args.delays_ms,
        repeats=args.repeats,
        warmups=args.warmups,
        profile=args.profile,
        order_seed=args.order_seed,
      )
    )
  )


if __name__ == "__main__":
  main()
