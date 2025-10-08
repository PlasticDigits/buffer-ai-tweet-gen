"""Tests for tweet_generator helper functions."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

import tweet_generator as tg


def _temp_image_path(tmp_path: Path, data: bytes = b"stub") -> Path:
    path = tmp_path / "image.png"
    path.write_bytes(data)
    return path


def test_append_tweet_index_appends_entry(tmp_path: Path) -> None:
    json_path = tmp_path / "tweet_output_1.json"
    image_path = tmp_path / "tweet_image_1.jpg"
    json_path.write_text("{}", encoding="utf-8")
    image_path.write_text("", encoding="utf-8")

    tg._append_tweet_index(
        output_dir=tmp_path,
        tweet="hello world",
        json_path=json_path,
        image_path=image_path,
    )

    index_path = tmp_path / "tweets.txt"
    content = index_path.read_text(encoding="utf-8")
    assert "hello world" in content
    assert "tweet_output_1.json" in content
    assert "tweet_image_1.jpg" in content


def test_payload_without_type_removes_type_key() -> None:
    payload = {"type": "text", "prompt": "hello", "other": 42}
    result = tg._payload_without_type(payload)
    assert "type" not in result
    assert result == {"prompt": "hello", "other": 42}


def test_coerce_text_output_handles_list_bytes() -> None:
    output = [b"Hello", b" ", "world"]
    assert tg._coerce_text_output(output) == "Hello world"


def test_persist_image_output_saves_file_like(tmp_path: Path) -> None:
    file_like = io.BytesIO(b"fake image data")
    path = tg._persist_image_output(
        file_like,
        output_dir=tmp_path,
        prefix="out",
        default_suffix=".png",
    )
    assert path.exists()
    assert path.read_bytes() == b"fake image data"


def test_persist_image_output_downloads_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    url = "https://example.com/image.png"

    class DummyResponse:
        def __enter__(self) -> "DummyResponse":
            return self

        def __exit__(self, *args: Any) -> None:  # pragma: no cover - nothing to do
            return None

        def read(self) -> bytes:
            return b"downloaded"

    monkeypatch.setattr(tg, "urlopen", lambda _: DummyResponse())
    path = tg._persist_image_output(
        url,
        output_dir=tmp_path,
        prefix="img",
        default_suffix=".jpg",
    )
    assert path.exists()
    assert path.read_bytes() == b"downloaded"


def test_open_binary_file_rejects_missing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        tg._open_binary_file(tmp_path / "missing.png")


def test_open_binary_file_opens_existing(tmp_path: Path) -> None:
    image = _temp_image_path(tmp_path)
    handle = tg._open_binary_file(image)
    try:
        assert handle.read() == b"stub"
    finally:
        handle.close()


