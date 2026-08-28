"""Release-content guard for files that must remain outside the repository."""

from __future__ import annotations

import argparse
from pathlib import Path


FORBIDDEN_SUFFIXES = {".engine", ".onnx", ".rgb", ".jpg", ".jpeg", ".png"}


def forbidden_paths(paths: list[Path]) -> list[Path]:
  return [path for path in paths if path.suffix.lower() in FORBIDDEN_SUFFIXES]


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("paths", nargs="+", type=Path)
  args = parser.parse_args()
  blocked = forbidden_paths(args.paths)
  if blocked:
    parser.error("forbidden release content: " + ", ".join(str(path) for path in blocked))


if __name__ == "__main__":
  main()
