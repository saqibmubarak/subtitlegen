from pathlib import Path
from typing import Any
from urllib.error import URLError

import pytest

from subtitlegen.domain.models import Cue
from subtitlegen.profiles.builder import AutomaticProfileBuilder
from subtitlegen.profiles.composer import ProfileComposer, safe_normalization_flags
from subtitlegen.profiles.extraction import LocalTranscriptExtractor
from subtitlegen.profiles.http import UrllibHttpClient
from subtitlegen.profiles.identity import MediaIdentity
from subtitlegen.profiles.models import GlossaryEntry, SeriesProfile
from subtitlegen.profiles.repository import ProfileRepository
from subtitlegen.profiles.resolver import ProfileResolver
from subtitlegen.profiles.websearch import WebSearchGlossarySource
from subtitlegen.profiles.wikipedia import WikipediaGlossarySource


class FakeHttp:
    def __init__(
        self,
        json_payloads: dict[str, Any],
        text_payloads: dict[str, str] | None = None,
    ) -> None:
        self.json_payloads = json_payloads
        self.text_payloads = text_payloads or {}

    def get_json(self, url: str) -> Any:
        for key, payload in self.json_payloads.items():
            if key in url:
                return payload
        raise RuntimeError(f"unexpected URL {url}")

    def get_text(self, url: str) -> str:
        for key, payload in self.text_payloads.items():
            if key in url:
                return payload
        raise RuntimeError(f"unexpected URL {url}")


def test_http_client_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError):
        UrllibHttpClient(timeout=0)
    with pytest.raises(ValueError):
        UrllibHttpClient(user_agent=" ")


def test_http_client_decodes_json_and_wraps_failures(monkeypatch: Any) -> None:
    class FakeResponse:
        headers = type("Headers", (), {"get_content_charset": staticmethod(lambda: "utf-8")})()

        def read(self) -> bytes:
            return b'{"ok": true}'

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(
        "subtitlegen.profiles.http.urlopen",
        lambda *_args, **_kwargs: FakeResponse(),
    )
    assert UrllibHttpClient().get_json("https://example.test") == {"ok": True}

    def boom(*_args: object, **_kwargs: object) -> None:
        raise URLError("offline")

    monkeypatch.setattr("subtitlegen.profiles.http.urlopen", boom)
    with pytest.raises(RuntimeError, match="HTTP request failed"):
        UrllibHttpClient().get_text("https://example.test")


def test_safe_flags_keep_single_english_words_prompt_only() -> None:
    assert safe_normalization_flags("Law") == (False, False)
    assert safe_normalization_flags("Kin'emon") == (True, True)
    assert safe_normalization_flags("Monkey D. Luffy") == (True, True)


def test_wikipedia_source_returns_none_when_search_is_empty() -> None:
    source = WikipediaGlossarySource(FakeHttp({"opensearch": ["Missing", [], [], []]}))
    assert source.fetch("Missing") is None


def test_wikipedia_source_parses_names_and_anime_markers() -> None:
    http = FakeHttp(
        {
            "opensearch": ["One Piece", ["One Piece"], ["desc"], ["url"]],
            "list=search": {"query": {"search": [{"title": "List of One Piece characters"}]}},
            "titles=List": {
                "query": {
                    "pages": {
                        "2": {
                            "title": "List of One Piece characters",
                            "extract": "",
                            "categories": [],
                            "revisions": [
                                {
                                    "slots": {
                                        "main": {
                                            "*": "'''Monkey D. Luffy''' '''Roronoa Zoro'''"
                                        }
                                    }
                                }
                            ],
                        }
                    }
                }
            },
            "titles=One": {
                "query": {
                    "pages": {
                        "1": {
                            "title": "One Piece",
                            "extract": "One Piece follows Monkey D. Luffy.",
                            "categories": [{"title": "Category:1997 manga"}],
                            "langlinks": [{"lang": "ja", "*": "ONE PIECE"}],
                            "revisions": [
                                {
                                    "slots": {
                                        "main": {
                                            "*": "| protagonist = [[Monkey D. Luffy]]\n'''Nami'''"
                                        }
                                    }
                                }
                            ],
                        }
                    }
                }
            },
        }
    )
    source = WikipediaGlossarySource(http)
    document = source.fetch("One Piece")
    assert document is not None
    assert document.japanese_title == "ONE PIECE"
    assert source.looks_like_anime(document)
    names = {entry.canonical for entry in source.terms(document)}
    assert {"Monkey D. Luffy", "Roronoa Zoro", "Nami"} <= names


def test_web_search_extracts_names_from_snippets() -> None:
    http = FakeHttp(
        {},
        {
            "duckduckgo": (
                '<a class="result__a">One Piece</a>'
                '<div class="result__snippet">Luffy and Trafalgar Law sail the Grand Line</div>'
            )
        },
    )
    terms = WebSearchGlossarySource(http).terms("One Piece")
    names = {entry.canonical for entry in terms}
    assert "Luffy" in names
    assert "Trafalgar Law" in names
    assert "Grand Line" in names


def test_builder_uses_search_when_wikipedia_is_thin() -> None:
    wikipedia = WikipediaGlossarySource(
        FakeHttp({"opensearch": ["X", [], [], []]})
    )

    class FakeSearch:
        def terms(self, title: str) -> tuple[GlossaryEntry, ...]:
            assert title == "Custom Show"
            return (GlossaryEntry("Aang", category="character"),)

    built = AutomaticProfileBuilder(wikipedia, FakeSearch()).build(  # type: ignore[arg-type]
        MediaIdentity("Custom Show", "custom-show")
    )
    assert built.source == "search"
    assert built.profile.terms[0].canonical == "Aang"
    assert not built.enable_visual


def test_transcript_extractor_merges_repeated_names() -> None:
    profile = SeriesProfile(1, "avatar", "Avatar", "en", ())
    cues = [
        Cue(0.0, 1.0, "Aang and Katara arrive"),
        Cue(1.0, 2.0, "Aang thanks Katara"),
        Cue(2.0, 3.0, "Okay then"),
    ]
    updated = LocalTranscriptExtractor().enrich(profile, cues)
    names = {entry.canonical for entry in updated.terms}
    assert names == {"Aang", "Katara"}


def test_composer_skips_conflicting_spellings() -> None:
    composer = ProfileComposer()
    profile = composer.compose(
        "Avatar",
        (
            (GlossaryEntry("Aang", aliases=("Ang",)),),
            (GlossaryEntry("Ang", aliases=("Aang",)),),
        ),
    )
    assert [entry.canonical for entry in profile.terms] == ["Aang"]


def test_resolver_prefers_shipped_then_cache_then_builder(tmp_path: Path) -> None:
    shipped = ProfileRepository(tmp_path / "shipped")
    shipped.save(SeriesProfile(1, "one-piece", "One Piece", "en", (GlossaryEntry("Luffy"),)))
    cache = ProfileRepository(tmp_path / "cache")
    video = tmp_path / "One Piece - 629.mkv"
    video.touch()

    class ExplodingBuilder:
        def build(self, identity: MediaIdentity) -> None:
            raise AssertionError(f"should not build {identity.title}")

    resolved = ProfileResolver(cache, shipped, builder=ExplodingBuilder()).resolve((video,))  # type: ignore[arg-type]
    assert resolved.source == "shipped"
    assert resolved.profile is not None
    assert resolved.profile.terms[0].canonical == "Luffy"

    unknown = tmp_path / "Custom Show - S01E01.mkv"
    unknown.touch()
    cache.save(SeriesProfile(1, "custom-show", "Custom Show", "en", (GlossaryEntry("Hero"),)))
    cached = ProfileResolver(cache, shipped, builder=ExplodingBuilder()).resolve((unknown,))  # type: ignore[arg-type]
    assert cached.source == "cache"
    assert cached.profile is not None
    assert cached.profile.terms[0].canonical == "Hero"


def test_resolver_maps_glossary_tokens_to_any_series_profile(tmp_path: Path) -> None:
    shipped = ProfileRepository(Path("profiles"))
    cache = ProfileRepository(tmp_path / "cache")

    class ExplodingBuilder:
        def build(self, identity: MediaIdentity) -> None:
            raise AssertionError(f"should not build {identity.title}")

    dressrosa = tmp_path / "[Group] Dressrosa 24.mp4"
    dressrosa.touch()
    one_piece = ProfileResolver(cache, shipped, builder=ExplodingBuilder()).resolve(
        (dressrosa,)
    )  # type: ignore[arg-type]
    assert one_piece.profile is not None
    assert one_piece.profile.profile_id == "one-piece"
    assert one_piece.identity is not None
    assert one_piece.identity.arc == "Dressrosa"

    ba_sing_se = tmp_path / "Ba Sing Se - S02E14.mp4"
    ba_sing_se.touch()
    avatar = ProfileResolver(cache, shipped, builder=ExplodingBuilder()).resolve(
        (ba_sing_se,)
    )  # type: ignore[arg-type]
    assert avatar.profile is not None
    assert avatar.profile.profile_id == "avatar"
    assert avatar.identity is not None
    assert avatar.identity.title == "Avatar"


def test_resolver_builds_and_writes_cache(tmp_path: Path) -> None:
    cache = ProfileRepository(tmp_path / "cache")
    video = tmp_path / "Made Up Series - S01E01.mkv"
    video.touch()

    class FakeBuilder:
        def build(self, identity: MediaIdentity) -> Any:
            from subtitlegen.profiles.builder import BuiltProfile

            profile = SeriesProfile(
                1,
                identity.profile_id,
                identity.title,
                "en",
                (GlossaryEntry("Hero"),),
            )
            return BuiltProfile(profile, enable_visual=True, source="wikipedia")

    resolved = ProfileResolver(cache, builder=FakeBuilder()).resolve((video,))  # type: ignore[arg-type]
    assert resolved.source == "wikipedia"
    assert resolved.enable_visual
    assert cache.load("made-up-series").terms[0].canonical == "Hero"


def test_builder_and_extractor_validate_thresholds() -> None:
    with pytest.raises(ValueError):
        AutomaticProfileBuilder(minimum_remote_terms=0)
    with pytest.raises(ValueError):
        LocalTranscriptExtractor(minimum_count=0)


def test_resolver_can_skip_automatic_creation(tmp_path: Path) -> None:
    video = tmp_path / "Unknown Show - S01E01.mkv"
    video.touch()
    resolved = ProfileResolver(ProfileRepository(tmp_path / "cache")).resolve(
        (video,),
        auto=False,
    )
    assert resolved.profile is None
    assert resolved.identity is not None
    assert resolved.source == "none"


def test_repository_round_trips_saved_profile(tmp_path: Path) -> None:
    repository = ProfileRepository(tmp_path)
    profile = SeriesProfile(
        1,
        "custom",
        "Custom",
        "en",
        (GlossaryEntry("Hero", aliases=("H3ro",), normalize_aliases=False),),
        (("日本語", "Japanese"),),
    )
    repository.save(profile)
    loaded = repository.load("custom")
    assert loaded == profile
