"""CPU-only Markdown summary for replay-run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(output_root: Path) -> str:
  rows: list[str] = []
  for summary_path in sorted(output_root.rglob("summary.json")):
    if summary_path.parent.name.startswith("warmup-"):
      continue
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    delay = data.get("transport_delay", {}).get("median_ms", "-")
    latency = data.get("inference_latency", {}).get("median_ms", "-")
    coverage = data.get("coverage", "-")
    rows.append(
      f"| {summary_path.parent.name} | {data.get('validity', '-')} | {data.get('outcome', '-')} | {delay} | {latency} | {coverage} |"
    )
  return "\n".join(
    [
      "# Replay run report",
      "",
      "| Run | Validity | Outcome | Median transport delay (ms) | Median inference latency (ms) | Coverage |",
      "| --- | --- | --- | ---: | ---: | ---: |",
      *rows,
      "",
    ]
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--runs", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(render(args.runs), encoding="utf-8")


if __name__ == "__main__":
  main()
