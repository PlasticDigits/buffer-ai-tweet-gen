#!/usr/bin/env python3
"""Tweet and image generation pipeline using Replicate models."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse
from urllib.request import urlopen

from dotenv import load_dotenv

import replicate

from prompt_builder import PromptTemplateError, render_prompt


REPO_ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = REPO_ROOT / "prompts"
MADLIB_DIR = PROMPTS_DIR / "madlib"
TEXT_PROMPT_PATH = PROMPTS_DIR / "gen-text-tweet.json"
IMAGE_TEXT_PROMPT_PATH = PROMPTS_DIR / "gen-text-imageprompt.json"
IMAGE_PROMPT_PATH = PROMPTS_DIR / "gen-image-tweet.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "replicate_tweet_outputs"

_MAX_OUTPUT_HISTORY = 1_000_000


def _ensure_env(var_name: str, *, fallback_names: tuple[str, ...] = ()) -> str:
    candidates = (var_name, *fallback_names)
    for name in candidates:
        value = os.getenv(name)
        if value:
            if name != var_name:
                os.environ[var_name] = value
            return value

    fallbacks_msg = (
        f" (also checked {', '.join(fallback_names)})" if fallback_names else ""
    )
    raise RuntimeError(
        f"Missing required environment variable: {var_name}{fallbacks_msg}."
        " Set it before running this script."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a tweet and companion image prompt via Replicate and "
            "save the results to a JSON file."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the output JSON and image will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional deterministic seed for madlib sampling.",
    )
    parser.add_argument(
        "--image-prefix",
        default="tweet_image",
        help="Filename prefix for generated image assets.",
    )
    parser.add_argument(
        "--json-prefix",
        default="tweet_output",
        help="Filename prefix for generated JSON summaries.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    load_dotenv()

    api_token = _ensure_env("REPLICATE_API_TOKEN", fallback_names=("REPLICATE_API_KEY",))
    text_model = _ensure_env("TEXT_MODEL")
    image_model = _ensure_env("IMAGE_MODEL")
    os.environ["REPLICATE_API_TOKEN"] = api_token

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = None
    if args.seed is not None:
        from random import Random

        rng = Random(args.seed)

    madlib_log: dict[str, list[str]] = {}

    try:
        tweet_payload = render_prompt(
            TEXT_PROMPT_PATH,
            madlib_dir=MADLIB_DIR,
            rng=rng,
            selection_log=madlib_log,
        )
        tweet_text = _run_text_model(text_model, tweet_payload).strip()
        if not tweet_text:
            raise RuntimeError("Text model returned empty tweet content.")

        image_prompt_payload = render_prompt(
            IMAGE_TEXT_PROMPT_PATH,
            variables={"tweet": tweet_text},
            madlib_dir=MADLIB_DIR,
            rng=rng,
            selection_log=madlib_log,
        )
        image_prompt = _run_text_model(text_model, image_prompt_payload).strip()
        if not image_prompt:
            raise RuntimeError("Image prompt generation returned empty prompt.")

        image_generation_payload = render_prompt(
            IMAGE_PROMPT_PATH,
            variables={"imageprompt": image_prompt},
            madlib_dir=MADLIB_DIR,
            rng=rng,
            selection_log=madlib_log,
        )

        image_path = _run_image_model(
            image_model,
            image_generation_payload,
            output_dir,
            prefix=args.image_prefix,
        )

    except PromptTemplateError as exc:
        print(f"Prompt templating failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Generation failed: {exc}", file=sys.stderr)
        return 1

    summary = {
        "tweet": tweet_text,
        "image": image_path.name,
        "image_prompt": image_prompt,
        "madlib": madlib_log,
    }
    summary_path = _next_output_path(
        output_dir, prefix=args.json_prefix, suffix=".json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _append_tweet_index(
        output_dir=output_dir,
        tweet=tweet_text,
        json_path=summary_path,
        image_path=image_path,
    )

    print(f"Tweet saved to {summary_path}")
    print(f"Image saved to {image_path}")
    return 0


def _run_text_model(model_id: str, payload: dict[str, Any]) -> str:
    input_payload = _payload_without_type(payload)
    output = replicate.run(model_id, input=input_payload)
    return _coerce_text_output(output)


def _run_image_model(
    model_id: str,
    payload: dict[str, Any],
    output_dir: Path,
    *,
    prefix: str,
) -> Path:
    input_payload = _payload_without_type(payload)

    image_inputs = input_payload.get("image_input")
    opened_files: list[Any] = []
    if image_inputs:
        file_handles: list[Any] = []
        for item in image_inputs:
            handle = _open_binary_file(item)
            opened_files.append(handle)
            file_handles.append(handle)
        input_payload["image_input"] = file_handles

    try:
        output = replicate.run(model_id, input=input_payload)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"Image model invocation failed: {exc}") from exc
    finally:
        for handle in opened_files:
            try:
                handle.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass

    try:
        return _persist_image_output(
            output, output_dir=output_dir, prefix=prefix, default_suffix=".jpg"
        )
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"No image content found in response: {output!r}") from exc


def _payload_without_type(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "type"}


def _coerce_text_output(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, Iterable) and not isinstance(output, (dict, bytes)):
        parts = []
        for item in output:
            if isinstance(item, (str, bytes)):
                parts.append(item.decode("utf-8") if isinstance(item, bytes) else item)
        return "".join(parts)
    if isinstance(output, dict):
        maybe_output = output.get("output") if "output" in output else None
        if isinstance(maybe_output, str):
            return maybe_output
        if isinstance(maybe_output, Iterable) and not isinstance(
            maybe_output, (dict, bytes)
        ):
            return "".join(str(part) for part in maybe_output)
    return str(output)


def _persist_image_output(
    output: Any, *, output_dir: Path, prefix: str, default_suffix: str
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(output, list):
        for idx, item in enumerate(output, start=1):
            try:
                return _persist_image_output(
                    item,
                    output_dir=output_dir,
                    prefix=f"{prefix}_{idx}",
                    default_suffix=default_suffix,
                )
            except PromptTemplateError:
                continue
        raise RuntimeError("Unable to persist image output from list response.")

    if hasattr(output, "read") and callable(output.read):
        return _save_file_like(output, output_dir, prefix, default_suffix)

    if isinstance(output, dict):
        for value in output.values():
            try:
                return _persist_image_output(
                    value,
                    output_dir=output_dir,
                    prefix=prefix,
                    default_suffix=default_suffix,
                )
            except RuntimeError:
                continue
        raise RuntimeError("Unable to persist image output from dictionary response.")

    if isinstance(output, str):
        if _looks_like_url(output):
            return _download_image(output, output_dir, prefix, default_suffix)
        raise RuntimeError(
            "Image model returned a string that is not a URL; cannot persist."
        )

    raise RuntimeError(f"Unsupported image output type: {type(output)!r}")


def _save_file_like(
    file_obj: Any, output_dir: Path, prefix: str, default_suffix: str
) -> Path:
    suffix = ""
    name = getattr(file_obj, "name", None)
    if isinstance(name, str) and name:
        suffix = Path(name).suffix
    if not suffix:
        suffix = default_suffix

    path = _next_output_path(output_dir, prefix=prefix, suffix=suffix)
    with path.open("wb") as handle:
        handle.write(file_obj.read())
    return path


def _download_image(url: str, output_dir: Path, prefix: str, default_suffix: str) -> Path:
    suffix = Path(urlparse(url).path).suffix or default_suffix
    path = _next_output_path(output_dir, prefix=prefix, suffix=suffix)

    with urlopen(url) as response:  # nosec - trusted output from Replicate models
        data = response.read()
    path.write_bytes(data)
    return path


def _next_output_path(output_dir: Path, *, prefix: str, suffix: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    for counter in range(_MAX_OUTPUT_HISTORY):
        candidate = output_dir / f"{prefix}_{timestamp}_{counter:04d}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(
        f"Unable to determine unique filename after {_MAX_OUTPUT_HISTORY} attempts."
    )


def _open_binary_file(path_like: Any):
    if isinstance(path_like, (str, os.PathLike)):
        resolved = Path(path_like)
    else:
        raise RuntimeError(f"Unsupported image_input entry: {path_like!r}")
    if not resolved.is_file():
        raise RuntimeError(f"Image input file not found: {resolved}")
    return resolved.open("rb")


def _looks_like_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _append_tweet_index(
    *,
    output_dir: Path,
    tweet: str,
    json_path: Path,
    image_path: Path,
) -> None:
    index_path = output_dir / "tweets.txt"
    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    entry_lines = [
        f"[{timestamp}]",
        tweet.strip(),
        f"JSON: {json_path.name}",
        f"Image: {image_path.name}",
        "",
    ]
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(entry_lines))
if __name__ == "__main__":
    sys.exit(main())


