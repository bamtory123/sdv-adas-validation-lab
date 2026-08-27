"""Collect only the runtime facts needed to reproduce a validation run."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def _version(package: str) -> str | None:
  try:
    return importlib.metadata.version(package)
  except importlib.metadata.PackageNotFoundError:
    return None


def _gpu() -> list[str]:
  try:
    result = subprocess.run(
      ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
      check=False,
      capture_output=True,
      text=True,
    )
  except OSError:
    return []
  return result.stdout.splitlines() if result.returncode == 0 else []


def collect() -> dict[str, object]:
  try:
    import onnxruntime as ort

    providers: list[str] = ort.get_available_providers()
  except ImportError:
    providers = []

  return {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "gpu": _gpu(),
    "packages": {name: _version(name) for name in ("onnx", "onnxruntime-gpu", "tensorrt")},
    "onnxruntime_providers": providers,
  }


def write(path: Path) -> None:
  path.write_text(json.dumps(collect(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
