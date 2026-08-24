"""Install third-party deps from pyproject.toml without the local package."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path


PADDLE_CPU = "paddlepaddle==3.3.1"
PADDLE_GPU = "paddlepaddle-gpu==3.3.1"


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


def parse_extras(args: list[str]) -> tuple[str, ...]:
    extras: list[str] = []
    for arg in args:
        extras.extend(part.strip() for part in arg.split(",") if part.strip())
    return tuple(extras)


def main() -> None:
    extras = parse_extras(sys.argv[1:])
    requirements = requirements_from_pyproject(Path("pyproject.toml"), extras)
    subprocess.check_call([sys.executable, "-m", "pip", "install", *requirements])


if __name__ == "__main__":
    main()
