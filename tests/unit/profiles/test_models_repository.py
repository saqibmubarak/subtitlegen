from pathlib import Path

import pytest

from subtitlegen.profiles.models import GlossaryEntry, SeriesProfile
from subtitlegen.profiles.repository import ProfileRepository


def test_glossary_entry_validates_and_filters_scope() -> None:
    entry = GlossaryEntry("Doflamingo", arcs=("Dressrosa",))
    assert entry.applies_to(arc="dressrosa")
    assert not entry.applies_to(arc="Wano")
    assert entry.applies_to()
    with pytest.raises(ValueError):
        GlossaryEntry(" ")
    with pytest.raises(ValueError):
        GlossaryEntry("Aang", aliases=("",))
    with pytest.raises(ValueError):
        GlossaryEntry("Aang", normalize_aliases="yes")  # type: ignore[arg-type]


def test_series_profile_rejects_schema_and_alias_collisions() -> None:
    with pytest.raises(ValueError):
        SeriesProfile(2, "avatar", "Avatar", "en", ())
    with pytest.raises(ValueError):
        SeriesProfile(
            1,
            "avatar",
            "Avatar",
            "en",
            (
                GlossaryEntry("Aang", aliases=("Avatar",)),
                GlossaryEntry("Avatar", aliases=()),
            ),
        )


def test_repository_loads_versioned_profiles() -> None:
    repository = ProfileRepository(Path("profiles"))
    assert repository.available() == ("avatar", "one-piece")
    avatar = repository.load("avatar")
    assert avatar.profile_id == "avatar"
    assert any(entry.canonical == "Aang" for entry in avatar.terms)
    with pytest.raises(FileNotFoundError):
        repository.load("missing")
    with pytest.raises(ValueError):
        repository.load("../avatar")
    assert ProfileRepository.default((Path("missing"), Path("profiles"))).load(
        "avatar"
    ).title == "Avatar"


def test_repository_rejects_malformed_yaml(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("terms: not-a-list", encoding="utf-8")
    with pytest.raises(RuntimeError):
        ProfileRepository(tmp_path).load("bad")
    (tmp_path / "scalar.yaml").write_text(
        """
schema_version: 1
profile_id: scalar
title: Scalar
language: en
terms:
  - canonical: Aang
    aliases: Ang
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError):
        ProfileRepository(tmp_path).load("scalar")
