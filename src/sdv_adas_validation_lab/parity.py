"""Semantic-output parity KPI for two runtimes on identical replay frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time_ns
from typing import Protocol

import numpy as np

from .contract import SensorFrame
from .replay import ReplaySource
from .runtime import OnnxReferenceRuntime, Prediction
from .tensorrt_runtime import TensorRtRuntime
from .preflight import collect


class RuntimeAdapter(Protocol):
  def infer(self, frame: SensorFrame) -> Prediction: ...


@dataclass(frozen=True, slots=True)
class ParitySummary:
  frames: int
  pixels: int
  matching_pixels: int
  label_agreement: float


def compare(source: ReplaySource, reference: RuntimeAdapter, candidate: RuntimeAdapter) -> ParitySummary:
  frames = pixels = matching = 0
  for frame in source:
    expected = reference.infer(frame).primary_labels
    observed = candidate.infer(frame).primary_labels
    if expected is None or observed is None or expected.shape != observed.shape:
      raise ValueError(f"runtime output labels are incompatible for frame {frame.frame_id}")
    frames += 1
    pixels += expected.size
    matching += int(np.count_nonzero(expected == observed))
  return ParitySummary(frames, pixels, matching, matching / pixels if pixels else 0.0)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def write_run(
  output_dir: Path,
  summary: ParitySummary,
  *,
  replay_manifest: Path,
  onnx_model: Path,
  tensorrt_engine: Path,
  minimum_agreement: float,
) -> dict[str, object]:
  """Write a self-contained parity result; model/engine content remains external."""
  output_dir.mkdir(parents=True, exist_ok=False)
  outcome = "pass" if summary.label_agreement >= minimum_agreement else "fail"
  result = {"validity": "valid", "outcome": outcome, "minimum_label_agreement": minimum_agreement, **asdict(summary)}
  manifest = {
    "kind": "runtime_parity/v1",
    "created_wall_ns": time_ns(),
    "preflight": collect(),
    "inputs": {
      "replay_manifest_sha256": _sha256(replay_manifest),
      "onnx_model_sha256": _sha256(onnx_model),
      "tensorrt_engine_sha256": _sha256(tensorrt_engine),
    },
  }
  (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  with (output_dir / "events.jsonl").open("w", encoding="utf-8") as output:
    for state in ("PROCESS_START", "MEASURE", "COLLECT"):
      output.write(json.dumps({"state": state, "wall_ns": time_ns()}) + "\n")
  return result


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--replay", required=True, type=Path)
  parser.add_argument("--onnx-model", required=True, type=Path)
  parser.add_argument("--tensorrt-engine", required=True, type=Path)
  parser.add_argument("--profile", default="fcn_resnet50_voc")
  parser.add_argument("--output", type=Path, help="summary JSON path (legacy single-file output)")
  parser.add_argument("--run-dir", type=Path, help="new isolated run artifact directory")
  parser.add_argument("--minimum-agreement", type=float, default=0.999)
  args = parser.parse_args()
  summary = compare(
    ReplaySource.from_jsonl(args.replay),
    OnnxReferenceRuntime(args.onnx_model, profile=args.profile),
    TensorRtRuntime(args.tensorrt_engine, profile=args.profile),
  )
  if not args.output and not args.run_dir:
    parser.error("one of --output or --run-dir is required")
  if args.output:
    args.output.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
  result = (
    write_run(
      args.run_dir,
      summary,
      replay_manifest=args.replay,
      onnx_model=args.onnx_model,
      tensorrt_engine=args.tensorrt_engine,
      minimum_agreement=args.minimum_agreement,
    )
    if args.run_dir
    else asdict(summary)
  )
  print(json.dumps(result))


if __name__ == "__main__":
  main()
