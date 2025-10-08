"""Prompt templating utilities for CL8Y content generation."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping


MADLIB_PREFIX = "madlib"
VARIABLE_PREFIX = "var"
PLACEHOLDER_PATTERN = re.compile(r"\$\{([^}:]+):([^}]+)\}")
DEFAULT_MADLIB_SUBDIR = "madlib"

_TEXT_SETTING_KEYS = {"system_prompt", "reasoning_effort", "verbosity"}
_IMAGE_SETTING_KEYS = {"aspect_ratio", "output_format", "image_input"}


class PromptTemplateError(RuntimeError):
    """Raised when templating fails due to configuration or data issues."""


@dataclass(frozen=True)
class _ResolutionContext:
    madlib_dir: Path
    variables: Mapping[str, Any]
    rng: random.Random
    selections: MutableMapping[str, list[str]]


def render_prompt(
    template_path: str | Path,
    *,
    variables: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    text_settings: Mapping[str, Any] | None = None,
    image_settings: Mapping[str, Any] | None = None,
    madlib_dir: str | Path | None = None,
    rng: random.Random | None = None,
    selection_log: MutableMapping[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Render a prompt template applying madlib substitutions and overrides.

    Args:
        template_path: Path to a JSON template file.
        variables: Values that can be referenced via ``${var:key}`` placeholders.
        overrides: Arbitrary top-level overrides applied before resolution.
        text_settings: Additional overrides allowed for text templates
            (fields: ``system_prompt``, ``reasoning_effort``, ``verbosity``).
        image_settings: Additional overrides allowed for image templates
            (fields: ``aspect_ratio``, ``output_format``, ``image_input``).
        madlib_dir: Directory containing madlib JSON files. Defaults to the
            ``madlib`` subdirectory next to the template.
        rng: Optional ``random.Random`` instance for deterministic sampling.

    Returns:
        A dictionary representing the fully rendered template payload.

    Raises:
        PromptTemplateError: If placeholders reference unknown data or files.
    """

    template_path = Path(template_path)
    if not template_path.exists():
        raise PromptTemplateError(f"Template not found: {template_path}")

    template_data = _load_json(template_path)

    resolved: dict[str, Any] = dict(template_data)

    if overrides:
        resolved.update(overrides)

    template_type = resolved.get("type")
    if template_type == "text" and text_settings:
        _validate_keys(text_settings, _TEXT_SETTING_KEYS, "text_settings")
        resolved.update(text_settings)
    if template_type == "image" and image_settings:
        _validate_keys(image_settings, _IMAGE_SETTING_KEYS, "image_settings")
        resolved.update(image_settings)

    context = _ResolutionContext(
        madlib_dir=_determine_madlib_dir(template_path, madlib_dir),
        variables=variables or {},
        rng=rng or random.Random(),
        selections=selection_log if selection_log is not None else {},
    )

    return _resolve_structure(resolved, context)


def _determine_madlib_dir(
    template_path: Path, explicit_dir: str | Path | None
) -> Path:
    if explicit_dir is not None:
        directory = Path(explicit_dir)
    else:
        directory = template_path.parent / DEFAULT_MADLIB_SUBDIR

    if not directory.exists():
        raise PromptTemplateError(f"Madlib directory not found: {directory}")
    if not directory.is_dir():
        raise PromptTemplateError(
            f"Madlib path is not a directory: {directory}"
        )
    return directory.resolve()


def _resolve_structure(value: Any, context: _ResolutionContext) -> Any:
    if isinstance(value, str):
        return _resolve_string(value, context)
    if isinstance(value, list):
        return [_resolve_structure(item, context) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_structure(item, context) for item in value)
    if isinstance(value, set):
        return {_resolve_structure(item, context) for item in value}
    if isinstance(value, dict):
        return {
            key: _resolve_structure(item, context)
            for key, item in value.items()
        }
    return value


def _resolve_string(source: str, context: _ResolutionContext) -> str:
    def replacer(match: re.Match[str]) -> str:
        prefix, key = match.group(1), match.group(2)
        if prefix == MADLIB_PREFIX:
            return _select_madlib_value(context, key)
        if prefix == VARIABLE_PREFIX:
            return _lookup_variable(context.variables, key)
        raise PromptTemplateError(
            f"Unsupported placeholder prefix '{prefix}' in '{source}'"
        )

    return PLACEHOLDER_PATTERN.sub(replacer, source)


def _lookup_variable(variables: Mapping[str, Any], key: str) -> str:
    if key not in variables:
        raise PromptTemplateError(
            f"Missing runtime variable '{key}' for placeholder '${{{VARIABLE_PREFIX}:{key}}}'"
        )
    value = variables[key]
    if isinstance(value, (str, int, float)):
        return str(value)
    raise PromptTemplateError(
        f"Variable '{key}' must resolve to a string-compatible value, got {type(value)!r}"
    )


def _select_madlib_value(context: _ResolutionContext, key: str) -> str:
    file_name = key if key.endswith(".json") else f"{key}.json"
    madlib_file = context.madlib_dir / file_name
    choices = _load_madlib_choices(madlib_file)
    try:
        selection = context.rng.choice(choices)
    except IndexError as exc:  # pragma: no cover - defensive guard
        raise PromptTemplateError(
            f"Madlib file '{madlib_file}' does not contain any entries"
        ) from exc

    context.selections.setdefault(key, []).append(selection)
    return selection


@lru_cache(maxsize=256)
def _load_madlib_choices(path: Path) -> tuple[str, ...]:
    if not path.exists():
        raise PromptTemplateError(f"Madlib file not found: {path}")
    raw = _load_json(path)
    if not isinstance(raw, Iterable) or isinstance(raw, (dict, str, bytes)):
        raise PromptTemplateError(
            f"Madlib file '{path}' must contain a JSON array of strings"
        )

    choices: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise PromptTemplateError(
                f"Madlib file '{path}' contains non-string entry: {item!r}"
            )
        cleaned = item.strip()
        if cleaned:
            choices.append(cleaned)

    if not choices:
        raise PromptTemplateError(
            f"Madlib file '{path}' does not contain any usable string entries"
        )

    return tuple(choices)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        raise PromptTemplateError(f"Invalid JSON in {path}: {exc}") from exc


def _validate_keys(
    provided: Mapping[str, Any], allowed: set[str], label: str
) -> None:
    invalid = set(provided) - allowed
    if invalid:
        names = ", ".join(sorted(invalid))
        raise PromptTemplateError(
            f"Unsupported key(s) for {label}: {names}. Allowed: {sorted(allowed)}"
        )


