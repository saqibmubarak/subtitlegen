import re
from collections.abc import Sequence
from statistics import fmean


def word_error_rate(reference: str, hypothesis: str) -> float:
    expected = _tokens(reference)
    actual = _tokens(hypothesis)
    if not expected:
        return 0.0 if not actual else 1.0
    previous = list(range(len(actual) + 1))
    for row, expected_word in enumerate(expected, start=1):
        current = [row]
        for column, actual_word in enumerate(actual, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected_word != actual_word),
                )
            )
        previous = current
    return previous[-1] / len(expected)


def character_error_rate(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    previous = list(range(len(hypothesis) + 1))
    for row, expected_character in enumerate(reference, start=1):
        current = [row]
        for column, actual_character in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected_character != actual_character),
                )
            )
        previous = current
    return previous[-1] / len(reference)


def terminology_recall(expected_terms: Sequence[str], hypothesis: str) -> float:
    if not expected_terms:
        return 1.0
    normalized = hypothesis.casefold()
    matched = sum(
        bool(re.search(rf"(?<!\w){re.escape(term.casefold())}(?!\w)", normalized))
        for term in expected_terms
    )
    return matched / len(expected_terms)


def mean_timestamp_error(reference: Sequence[float], actual: Sequence[float]) -> float:
    if len(reference) != len(actual):
        raise ValueError("timestamp sequences must have equal lengths")
    if not reference:
        return 0.0
    return fmean(
        abs(expected - observed)
        for expected, observed in zip(reference, actual, strict=True)
    )


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.casefold())
