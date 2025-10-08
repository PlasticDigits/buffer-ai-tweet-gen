#!/usr/bin/env python3
"""Utility script to run the Replicate text model defined in the TEXT_MODEL env var."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Iterable

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
        f"Missing required environment variable: {var_name}{fallbacks_msg}. "
        f"Set it before running this script."
    )


def _is_file_like(value: Any) -> bool:
    return hasattr(value, "read") and callable(value.read)


def _print_text_chunk(chunk: str | bytes) -> None:
    if isinstance(chunk, bytes):
        chunk = chunk.decode("utf-8", errors="replace")
    sys.stdout.write(chunk)
    sys.stdout.flush()


def _save_file_output(file_obj: Any, output_dir: Path, prefix: str, index: int) -> Path:
    suffix = ""
    name = getattr(file_obj, "name", None)
    if isinstance(name, str) and name:
        suffix = Path(name).suffix
    if not suffix:
        suffix = ".bin"

    output_path = output_dir / f"{prefix}_{index}{suffix}"
    with output_path.open("wb") as f:
        f.write(file_obj.read())
    return output_path



def _collect_file_outputs_from_event(event: Any) -> list[Any]:
    candidates: list[Any] = []

    for attr in ("output", "data", "delta"):
        if not hasattr(event, attr):
            continue
        value = getattr(event, attr)
        if value is None:
            continue
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
            candidates.extend(value)
        else:
            candidates.append(value)

    file_outputs: list[Any] = []
    for candidate in candidates:
        if _is_file_like(candidate):
            file_outputs.append(candidate)
        elif isinstance(candidate, dict):
            for nested_value in candidate.values():
                if _is_file_like(nested_value):
                    file_outputs.append(nested_value)
    return file_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Replicate text model configured in the TEXT_MODEL env var."
    )
    parser.add_argument(
        "--prompt",
        default="Hello from Replicate!",
        help="Prompt to send to the model.",
    )
    parser.add_argument(
        "--output-dir",
        default="replicate_outputs",
        help="Directory to store any file outputs returned by the model.",
    )
    parser.add_argument(
        "--file-prefix",
        default="replicate_output",
        help="Filename prefix for saved file outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    load_dotenv()

    api_token = _ensure_env("REPLICATE_API_TOKEN", fallback_names=("REPLICATE_API_KEY",))
    model_identifier = _ensure_env("TEXT_MODEL")
    os.environ["REPLICATE_API_TOKEN"] = api_token

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming Replicate model '{model_identifier}' with provided prompt...")
    print("")

    saved_files: list[Path] = []

    try:
        stream = replicate.stream(
            model_identifier,
            input={"prompt": args.prompt},
        )
        for index, event in enumerate(stream, start=1):
            print(str(event), end="", flush=True)
            for file_idx, file_output in enumerate(
                _collect_file_outputs_from_event(event), start=1
            ):
                saved_path = _save_file_output(
                    file_output,
                    output_dir,
                    args.file_prefix,
                    index * 1000 + file_idx,
                )
                saved_files.append(saved_path)
                print(f"\nSaved file output to {saved_path}")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error while streaming model: {exc}")
        return 1

    if saved_files:
        print(f"\nSaved {len(saved_files)} file output(s).")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

