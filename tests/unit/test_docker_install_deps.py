import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "docker_install_deps.py"
_SPEC = importlib.util.spec_from_file_location("docker_install_deps", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_DEPS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DEPS)


def test_parse_extras_splits_commas_and_skips_blanks() -> None:
    assert _DEPS.parse_extras(["cuda,ocr", "dev", ""]) == ("cuda", "ocr", "dev")


def test_cuda_ocr_swaps_cpu_paddle_for_gpu_wheel() -> None:
    reqs = _DEPS.requirements_from_pyproject(Path("pyproject.toml"), ("cuda", "ocr"))
    assert "paddlepaddle-gpu==3.3.1" in reqs
    assert "paddlepaddle==3.3.1" not in reqs
    cpu = _DEPS.requirements_from_pyproject(Path("pyproject.toml"), ("ocr",))
    assert "paddlepaddle==3.3.1" in cpu
    assert "paddlepaddle-gpu==3.3.1" not in cpu


def test_requirements_include_core_and_named_extras() -> None:
    reqs = _DEPS.requirements_from_pyproject(Path("pyproject.toml"), ("nemo",))
    assert "faster-whisper==1.2.1" in reqs
    assert "nemo_toolkit[asr]==3.0.0" in reqs
    assert "paddleocr==3.7.0" not in reqs


def test_dockerfile_extras_layer_ignores_readme_and_src() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    extras, marker, rest = dockerfile.partition("python docker_install_deps.py")
    assert marker
    assert "COPY pyproject.toml" in extras
    assert "docker_install_deps.py" in extras
    assert "type=bind" not in extras
    assert "ReadMe.md" not in extras
    assert "COPY src" not in extras
    assert "ReadMe.md" in rest
    assert "COPY src" in rest


def test_windows_titles_reuses_parakeet_srt() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    dialogue, _, titles = compose.partition("windows-titles:")
    assert "SUBTITLEGEN_PRESET: english-fast" in dialogue
    assert "--reuse-srt" in titles
    assert "--no-visual-text" in dialogue
    assert "TZ: ${TZ:-}" in compose
