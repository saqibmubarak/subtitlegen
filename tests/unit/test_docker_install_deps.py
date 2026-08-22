import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "docker_install_deps.py"
_SPEC = importlib.util.spec_from_file_location("docker_install_deps", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_DEPS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DEPS)


def test_parse_extras_splits_commas_and_skips_blanks() -> None:
    assert _DEPS.parse_extras(["cuda,ocr", "dev", ""]) == ("cuda", "ocr", "dev")


def test_requirements_include_core_and_named_extras() -> None:
    reqs = _DEPS.requirements_from_pyproject(Path("pyproject.toml"), ("nemo",))
    assert "faster-whisper==1.2.1" in reqs
    assert "nemo_toolkit[asr]==3.0.0" in reqs
    assert "paddleocr==3.7.0" not in reqs
