"""Build an unpublished, redacted portfolio release bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from .evidence import render_evidence
from .release_audit import forbidden_paths


FILES = ("README.md", "docs/release-notes-v0.1.0-portfolio.md", "docs/release-readiness.md", "docs/model-fcn-resnet50.md")


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def build(repository_root: Path, output_dir: Path) -> dict[str, object]:
  output_dir.mkdir(parents=True, exist_ok=False)
  for relative in FILES:
    source = repository_root / relative
    target = output_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
  sample_root = repository_root / "samples" / "evidence"
  (output_dir / "evidence-sample.md").write_text(
    render_evidence([("redacted-sample", sample_root / "ground-truth.json")], [sample_root / "replay"]), encoding="utf-8"
  )
  blocked = forbidden_paths([path for path in output_dir.rglob("*") if path.is_file()])
  if blocked:
    raise ValueError("release bundle contains forbidden content")
  try:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository_root, check=True, capture_output=True, text=True).stdout.strip()
  except (OSError, subprocess.CalledProcessError):
    commit = None
  manifest = {"kind": "portfolio_release_bundle/v1", "git_commit": commit, "files": {str(path.relative_to(output_dir)): _sha256(path) for path in output_dir.rglob("*") if path.is_file()}}
  (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  return manifest


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repository-root", type=Path, default=Path("."))
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  print(json.dumps(build(args.repository_root, args.output), sort_keys=True))


if __name__ == "__main__":
  main()
