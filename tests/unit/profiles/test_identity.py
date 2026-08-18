from pathlib import Path

import pytest

from subtitlegen.profiles.identity import MediaIdentity, PathIdentityInferencer, slugify


def test_media_identity_rejects_blank_fields() -> None:
    with pytest.raises(ValueError):
        MediaIdentity(" ", "blank")


def test_slugify_normalizes_punctuation() -> None:
    assert slugify("One Piece") == "one-piece"
    assert slugify("Avatar: The Last Airbender") == "avatar-the-last-airbender"


def test_inferencer_reads_season_episode_and_arc(tmp_path: Path) -> None:
    inferencer = PathIdentityInferencer()
    video = tmp_path / "One Piece" / "One Piece - S07E629 - Dressrosa.mkv"
    video.parent.mkdir()
    video.touch()
    identity = inferencer.infer(video)
    assert identity is not None
    assert identity.title == "One Piece"
    assert identity.profile_id == "one-piece"
    assert identity.episode == "629"
    assert identity.arc == "Dressrosa"


def test_inferencer_uses_parent_folder_and_ignores_junk_names(tmp_path: Path) -> None:
    inferencer = PathIdentityInferencer()
    video = tmp_path / "Avatar The Last Airbender" / "episode.mp4"
    video.parent.mkdir()
    video.touch()
    identity = inferencer.infer(video)
    assert identity is not None
    assert identity.title == "Avatar The Last Airbender"

    junk = tmp_path / "videos" / "video.mp4"
    junk.parent.mkdir()
    junk.touch()
    assert inferencer.infer(junk) is None
    assert inferencer.infer(tmp_path / "test_cli_case" / "clip.mp4") is None


def test_inferencer_reads_directory_title(tmp_path: Path) -> None:
    series = tmp_path / "AttackOnTitan"
    series.mkdir()
    identity = PathIdentityInferencer().infer(series)
    assert identity is not None
    assert identity.title == "Attack On Titan"
    assert identity.profile_id == "attack-on-titan"
