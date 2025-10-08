"""Tests for prompt_builder module."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from prompt_builder import PromptTemplateError, render_prompt


@pytest.fixture(name="tmp_prompts")
def fixture_tmp_prompts(tmp_path: Path) -> Path:
    base = tmp_path / "prompts"
    madlib_dir = base / "madlib"
    madlib_dir.mkdir(parents=True)

    (madlib_dir / "topic.json").write_text(
        json.dumps(["Topic A", "Topic B"]), encoding="utf-8"
    )
    (madlib_dir / "mood.json").write_text(
        json.dumps(["Calm", "Hype"]), encoding="utf-8"
    )
    (madlib_dir / "scene.json").write_text(
        json.dumps(["City", "Forest"]), encoding="utf-8"
    )

    (base / "gen-text-tweet.json").write_text(
        json.dumps(
            {
                "type": "text",
                "prompt": "Topic: ${madlib:topic}",
                "system_prompt": "default",
            }
        ),
        encoding="utf-8",
    )

    (base / "gen-text-imageprompt.json").write_text(
        json.dumps(
            {
                "type": "text",
                "prompt": (
                    "Tweet: ${var:tweet}\nMood: ${madlib:mood}\nScene: ${madlib:scene}"
                ),
            }
        ),
        encoding="utf-8",
    )

    (base / "gen-image-tweet.json").write_text(
        json.dumps(
            {
                "type": "image",
                "prompt": "${var:imageprompt}",
                "aspect_ratio": "1:1",
            }
        ),
        encoding="utf-8",
    )

    return base


def test_render_text_template_with_madlib(tmp_prompts: Path) -> None:
    rng = random.Random(123)
    result = render_prompt(tmp_prompts / "gen-text-tweet.json", rng=rng)

    assert result["type"] == "text"
    assert result["system_prompt"] == "default"
    assert result["prompt"].startswith("Topic: ")
    assert result["prompt"].split(": ", maxsplit=1)[1] in {"Topic A", "Topic B"}


def test_render_with_variables_and_text_settings(tmp_prompts: Path) -> None:
    rng = random.Random(42)
    context = {"tweet": "Mint live now"}
    overrides = {"system_prompt": "override"}
    text_settings = {"verbosity": "high"}

    result = render_prompt(
        tmp_prompts / "gen-text-imageprompt.json",
        variables=context,
        overrides=overrides,
        text_settings=text_settings,
        rng=rng,
    )

    assert result["type"] == "text"
    assert result["system_prompt"] == "override"
    assert result["verbosity"] == "high"
    assert "Mint live now" in result["prompt"]


def test_render_image_template_with_overrides(tmp_prompts: Path) -> None:
    rng = random.Random(9)
    variables = {"imageprompt": "Example prompt"}
    image_settings = {"aspect_ratio": "16:9"}

    result = render_prompt(
        tmp_prompts / "gen-image-tweet.json",
        variables=variables,
        image_settings=image_settings,
        rng=rng,
    )

    assert result["type"] == "image"
    assert result["aspect_ratio"] == "16:9"
    assert result["prompt"] == "Example prompt"


def test_missing_variable_raises(tmp_prompts: Path) -> None:
    with pytest.raises(PromptTemplateError):
        render_prompt(tmp_prompts / "gen-text-imageprompt.json")


def test_unknown_text_setting_key(tmp_prompts: Path) -> None:
    with pytest.raises(PromptTemplateError):
        render_prompt(
            tmp_prompts / "gen-text-imageprompt.json",
            text_settings={"unknown": "value"},
        )


