"""Combine ground-truth and replay artifacts into one portable Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .report import render


def render_evidence(evaluations: list[tuple[str, Path]], run_roots: list[Path]) -> str:
  lines = ["# Validation evidence", "", "## Ground-truth evaluation", "", "| Evaluation | Pixels | mIoU | Pixel accuracy |", "| --- | ---: | ---: | ---: |"]
  for name, path in evaluations:
    data = json.loads(path.read_text(encoding="utf-8"))
    lines.append(f"| {name} | {data['evaluated_pixels']} | {data['mean_iou']:.6f} | {data['pixel_accuracy']:.6f} |")
  for root in run_roots:
    lines.extend(["", f"## Replay runs: {root.name}", "", render(root).rstrip()])
  return "\n".join(lines) + "\n"


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--evaluation", action="append", nargs=2, metavar=("NAME", "SUMMARY_JSON"), default=[])
  parser.add_argument("--runs", action="append", type=Path, default=[])
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  if not args.evaluation and not args.runs:
    parser.error("provide at least one --evaluation or --runs")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(render_evidence([(name, Path(path)) for name, path in args.evaluation], args.runs), encoding="utf-8")


if __name__ == "__main__":
  main()
