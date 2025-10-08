#!/usr/bin/env python3
"""Test runner for the Replicate image model defined in the IMAGE_MODEL env var."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

import replicate


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


def _is_file_like(value: Any) -> bool:
    return hasattr(value, "read") and callable(value.read)


def _maybe_get_url(value: Any) -> str | None:
    candidate = getattr(value, "url", None)
    if callable(candidate):
        try:
            resolved = candidate()
            if isinstance(resolved, str) and resolved:
                return resolved
        except Exception:  # pragma: no cover - best effort helper
            return None
    elif isinstance(candidate, str) and candidate:
        return candidate
    return None


def _save_file_output(
    file_obj: Any, output_dir: Path, prefix: str, index: int, default_suffix: str
) -> Path:
    suffix = ""
    name = getattr(file_obj, "name", None)
    if isinstance(name, str) and name:
        suffix = Path(name).suffix
    if not suffix:
        suffix = default_suffix

    output_path = output_dir / f"{prefix}_{index}{suffix}"
    with output_path.open("wb") as file_handle:
        file_handle.write(file_obj.read())
    return output_path


def _handle_output(
    output: Any, output_dir: Path, file_prefix: str, default_suffix: str
) -> list[Path]:
    saved_paths: list[Path] = []

    def process(value: Any, idx: int) -> None:
        if value is None:
            return

        url = _maybe_get_url(value)
        if url:
            print(f"Output URL: {url}")

        if _is_file_like(value):
            saved_paths.append(
                _save_file_output(value, output_dir, file_prefix, idx, default_suffix)
            )
            print(f"Saved file output to {saved_paths[-1]}")
            return

        if isinstance(value, (list, tuple)):
            for inner_idx, item in enumerate(value, start=1):
                process(item, idx * 1000 + inner_idx)
            return

        if isinstance(value, dict):
            for inner_idx, item in enumerate(value.values(), start=1):
                process(item, idx * 1000 + inner_idx)
            return

        if isinstance(value, str):
            print(value)
            return

        print(value)

    if isinstance(output, list):
        for index, item in enumerate(output, start=1):
            process(item, index)
    else:
        process(output, 1)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Replicate image model configured in the IMAGE_MODEL env var."
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Attached image is CL8Y logo, to be included in scene.\n"
            "Night storm on a jagged coast: a chrome CL8Y logo seared into wet basalt radiates"
            " a gold encrypted ring as aqua/magenta light streaks rip through spray, with "
            "\"FORGED HASHES PERISH\" in bold.\n\n"
            "hype, antifragile, cyber-mythic, kinetic, high-contrast"
        ),
        help="Prompt to send to the model.",
    )
    parser.add_argument(
        "--aspect-ratio",
        default="16:9",
        choices=("16:9", "1:1", "4:5"),
        help="Aspect ratio preset for the generated image.",
    )
    parser.add_argument(
        "--image-input",
        nargs="*",
        default=("LOGO_CL8Y.png", "LOGO_CL8Y_CYBER.png"),
        help="One or more image file paths to include as input (omit to skip).",
    )
    parser.add_argument(
        "--output-format",
        default="jpg",
        choices=("jpg", "png", "webp"),
        help="Output file format.",
    )
    parser.add_argument(
        "--output-dir",
        default="replicate_image_outputs",
        help="Directory to store generated image files.",
    )
    parser.add_argument(
        "--file-prefix",
        default="replicate_image",
        help="Filename prefix for saved outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    load_dotenv(override=True)

    api_token = _ensure_env("REPLICATE_API_TOKEN", fallback_names=("REPLICATE_API_KEY",))
    model_identifier = _ensure_env("IMAGE_MODEL")
    os.environ["REPLICATE_API_TOKEN"] = api_token

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_input: list[Any] = []
    opened_files: list[Any] = []
    for image_path in args.image_input:
        if not image_path:
            continue
        path_obj = Path(image_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Image input file not found: {path_obj}")
        handle = path_obj.open("rb")
        opened_files.append(handle)
        image_input.append(handle)

    input_payload = {
        "prompt": args.prompt,
        "image_input": image_input,
        "aspect_ratio": args.aspect_ratio,
        "output_format": args.output_format,
    }

    print(f"Running Replicate image model '{model_identifier}'...")

    try:
        output = replicate.run(model_identifier, input=input_payload)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error while running model: {exc}")
        return 1
    finally:
        for handle in opened_files:
            try:
                handle.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass

    default_suffix = f".{args.output_format}"
    saved_paths = _handle_output(
        output,
        output_dir,
        args.file_prefix,
        default_suffix,
    )

    if saved_paths:
        print(f"Saved {len(saved_paths)} file output(s).")
    else:
        print("No file outputs were returned.")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

