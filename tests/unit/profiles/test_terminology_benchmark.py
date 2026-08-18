from pathlib import Path
from typing import Any

import yaml

from subtitlegen.profiles.normalizer import GlossaryNormalizer
from subtitlegen.profiles.repository import ProfileRepository


def test_committed_terminology_fixture_improves_exact_case_recall() -> None:
    cases: list[dict[str, Any]] = yaml.safe_load(
        Path("tests/fixtures/terminology_cases.yaml").read_text(encoding="utf-8")
    )
    repository = ProfileRepository(Path("profiles"))
    normalizer = GlossaryNormalizer()
    before = sum(case["input"] == case["expected"] for case in cases)
    outputs = [
        normalizer.normalize(case["input"], repository.load(case["profile"]))
        for case in cases
    ]
    after = sum(output == case["expected"] for output, case in zip(outputs, cases, strict=True))

    assert before == 1
    assert after == len(cases)
    assert outputs[-1] == cases[-1]["input"]


def test_dressrosa_annotation_records_name_improvement() -> None:
    annotations: dict[str, Any] = yaml.safe_load(
        Path("tests/fixtures/dressrosa_terminology_annotations.yaml").read_text(
            encoding="utf-8"
        )
    )
    terms = annotations["terms"]
    baseline = sum(item["baseline"] == item["expected"] for item in terms)
    profiled = sum(item["profiled"] == item["expected"] for item in terms)
    assert baseline == 6
    assert profiled == 9
