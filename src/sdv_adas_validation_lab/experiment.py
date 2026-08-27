"""Batch runtime/delay experiment orchestration with an aggregate manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .faults import FaultConfig
from .harness import run_repeated
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
) -> dict[str, object]:
  output_dir.mkdir(parents=True, exist_ok=False)
  runtime = TensorRtRuntime(engine_path, profile=profile)
  conditions: list[dict[str, object]] = []
  for delay_ms in delays_ms:
    condition_dir = output_dir / f"delay-{delay_ms:03d}ms"
    summaries = run_repeated(source, FaultConfig(delay_ms=delay_ms), condition_dir, repeats, runtime, warmups)
    conditions.append({"delay_ms": delay_ms, "measured_runs": len(summaries), "directory": condition_dir.name})
  manifest = {
    "kind": "delay-experiment/v1",
    "source_hash": source.content_hash,
    "engine_sha256": _sha256(engine_path),
    "profile": profile,
    "repeats": repeats,
    "warmups": warmups,
    "conditions": conditions,
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
      )
    )
  )


if __name__ == "__main__":
  main()
