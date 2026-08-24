from __future__ import annotations

from collections.abc import Callable, Sequence
from difflib import SequenceMatcher
from typing import Any, Protocol

from subtitlegen.errors import BackendOutOfMemoryError, BackendUnavailableError
from subtitlegen.profiles.correction import ConservativeLocalCorrector
from subtitlegen.profiles.models import SeriesProfile
from subtitlegen.profiles.normalizer import GlossaryNormalizer


class Translator(Protocol):
    def translate(self, text: str) -> str:
        """Translate source text locally."""


ModelFactory = Callable[[str, str], tuple[Any, Any]]


class NllbLocalTranslator:
    DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        profile: SeriesProfile | None = None,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._profile = profile
        self._model_factory = model_factory
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._cache: dict[str, str] = {}

    def translate(self, text: str) -> str:
        return self.translate_many((text,))[0]

    def translate_many(self, texts: Sequence[str]) -> list[str]:
        if not texts:
            return []
        results: list[str | None] = [None] * len(texts)
        pending: list[tuple[int, str]] = []
        for index, raw in enumerate(texts):
            source = raw.strip()
            if not source:
                raise ValueError("translation source text must not be blank")
            cached = self._cache.get(source)
            if cached is not None:
                results[index] = cached
                continue
            if self._profile is not None:
                known_translation = self._profile_translation(source)
                if known_translation is not None:
                    self._cache[source] = known_translation
                    results[index] = known_translation
                    continue
            pending.append((index, source))
        unique = list(dict.fromkeys(source for _index, source in pending))
        if unique:
            translated = self._generate_batch(unique)
            for source, result in zip(unique, translated, strict=True):
                self._cache[source] = result
            for index, source in pending:
                results[index] = self._cache[source]
        return [item if item is not None else "" for item in results]

    def _generate_batch(self, sources: list[str]) -> list[str]:
        tokenizer, model = self._load_model()
        try:
            inputs = tokenizer(sources, return_tensors="pt", padding=True)
            if self._device != "cpu":
                inputs = {key: value.to(self._device) for key, value in inputs.items()}
            translated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
                max_new_tokens=128,
            )
            decoded = [
                str(text).strip()
                for text in tokenizer.batch_decode(translated, skip_special_tokens=True)
            ]
        except RuntimeError as error:
            if "out of memory" in str(error).casefold():
                raise BackendOutOfMemoryError(
                    "NLLB exhausted memory; run translation on CPU after releasing ASR"
                ) from error
            raise
        if len(decoded) != len(sources) or any(not text for text in decoded):
            raise RuntimeError("NLLB returned an empty translation")
        if self._profile is None:
            return decoded
        normalizer = GlossaryNormalizer()
        corrector = ConservativeLocalCorrector()
        safe_terms = tuple(
            entry.canonical
            for entry in self._profile.terms
            if entry.normalize_aliases and entry.normalize_canonical
        )
        return [
            corrector.correct(normalizer.normalize(text, self._profile), glossary=safe_terms)
            for text in decoded
        ]

    def _profile_translation(self, source: str) -> str | None:
        if self._profile is None:
            return None
        translations = dict(self._profile.visual_translations)
        exact = translations.get(source)
        if exact is not None:
            return exact
        if len(source) < 6:
            return None
        ranked = sorted(
            (
                (SequenceMatcher(None, source, expected).ratio(), translation)
                for expected, translation in translations.items()
            ),
            reverse=True,
        )
        if not ranked or ranked[0][0] < 0.82:
            return None
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.1:
            return None
        return ranked[0][1]

    def close(self) -> None:
        self._tokenizer = None
        self._model = None
        self._cache.clear()

    def _load_model(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        factory = self._model_factory
        if factory is None:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as error:
                raise BackendUnavailableError(
                    "NLLB translation requires subtitlegen[ocr]"
                ) from error

            def factory(model_name: str, device: str) -> tuple[Any, Any]:
                tokenizer_class: Any = AutoTokenizer
                model_class: Any = AutoModelForSeq2SeqLM
                tokenizer = tokenizer_class.from_pretrained(
                    model_name,
                    src_lang="jpn_Jpan",
                    use_fast=False,
                )
                model = model_class.from_pretrained(model_name)
                if device != "cpu":
                    model = model.to(device)
                return tokenizer, model

        self._tokenizer, self._model = factory(self._model_name, self._device)
        return self._tokenizer, self._model
