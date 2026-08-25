"""Install third-party deps from pyproject.toml without the local package."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path


PADDLE_CPU = "paddlepaddle==3.3.1"
PADDLE_GPU = "paddlepaddle-gpu==3.3.1"
# PyPI only has paddlepaddle-gpu 2.x. 3.3.1 GPU wheels live on Paddle's index.
# Default matches the CUDA 12.9 PyTorch image in Dockerfile.
PADDLE_GPU_INDEX = os.environ.get(
    "SUBTITLEGEN_PADDLE_GPU_INDEX",
    "https://www.paddlepaddle.org.cn/packages/stable/cu129/",
)


def requirements_from_pyproject(pyproject: Path, extras: tuple[str, ...]) -> list[str]:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    requirements = list(data["project"]["dependencies"])
    optional = data["project"]["optional-dependencies"]
    for extra in extras:
        if extra not in optional:
            available = ", ".join(optional)
            raise SystemExit(f"unknown extra '{extra}'; available: {available}")
        requirements.extend(optional[extra])
    if "cuda" in extras and "ocr" in extras:
        requirements = [
            PADDLE_GPU if requirement == PADDLE_CPU else requirement
            for requirement in requirements
        ]
    return requirements


def pip_install_commands(requirements: list[str]) -> list[list[str]]:
    """PyPI for everything except the GPU wheel, which is only on Paddle's index."""
    python_pip = [sys.executable, "-m", "pip", "install"]
    if PADDLE_GPU not in requirements:
        return [python_pip + requirements]
    others = [item for item in requirements if item != PADDLE_GPU]
    return [
        python_pip + others,
        [*python_pip, PADDLE_GPU, "-i", PADDLE_GPU_INDEX],
    ]


def parse_extras(args: list[str]) -> tuple[str, ...]:
    extras: list[str] = []
    for arg in args:
        extras.extend(part.strip() for part in arg.split(",") if part.strip())
    return tuple(extras)


def main() -> None:
    extras = parse_extras(sys.argv[1:])
    requirements = requirements_from_pyproject(Path("pyproject.toml"), extras)
    for command in pip_install_commands(requirements):
        subprocess.check_call(command)


if __name__ == "__main__":
    main()
