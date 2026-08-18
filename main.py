import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

app = import_module("subtitlegen.cli").app


if __name__ == "__main__":
    app()
