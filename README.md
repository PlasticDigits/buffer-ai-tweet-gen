# Tweet Generator

Generate short-form content and companion imagery with `tweet_generator.py`, a Replicate-powered pipeline that assembles prompts from the `prompts/` directory, calls large language and diffusion models, and saves the results locally.

## Prerequisites

- Python 3.10 or newer.
- Replicate account and API access.
- Environment variables for both the text and image models you plan to call.

## Local Setup

1. Create the virtual environment (the project standard is `.venv`):
   ```bash
   python3 -m venv .venv
   ```
2. Activate it for your shell session:
   ```bash
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

To leave the environment later, run `deactivate`.

## Configure Environment Variables

The script reads configuration from your environment (optionally via a `.env` file loaded by `python-dotenv`). At minimum, set:

- `REPLICATE_API_TOKEN` (or `REPLICATE_API_KEY`): your Replicate API token.
- `TEXT_MODEL`: the Replicate model ID to generate tweet copy and the image prompt text.
- `IMAGE_MODEL`: the Replicate model ID that produces the image.

You can place these values in a `.env` file at the repository root:

```
REPLICATE_API_TOKEN=your-token
TEXT_MODEL=replicate/text-model-id
IMAGE_MODEL=replicate/image-model-id
```

Additional optional prompt variables can be configured in the JSON templates under `prompts/`.

## Run the Generator

Once your environment is activated and configured:

```bash
python tweet_generator.py \
  --output-dir replicate_tweet_outputs \
  --seed 1234 \
  --image-prefix tweet_image \
  --json-prefix tweet_output
```

All command-line flags are optional:

- `--output-dir`: folder for generated JSON summaries and image files (defaults to `replicate_tweet_outputs/`).
- `--seed`: deterministic seed for the prompt madlibs; omit for fully random output.
- `--image-prefix`: prefix for saved image filenames (default `tweet_image`).
- `--json-prefix`: prefix for saved JSON filenames (default `tweet_output`).

On success, the script prints the saved paths and writes a JSON summary containing the tweet text, the image prompt, the selected madlib fragments, and the image filename.

Each run also appends the generated tweet text to `tweets.txt` inside the output directory, alongside references to the corresponding JSON and image files for quick copy/paste.

## Outputs

- Image assets are saved under the output directory, e.g. `replicate_tweet_outputs/tweet_image_YYYYMMDDTHHMMSS_0000.jpg`.
- Metadata summaries land alongside the images with the specified JSON prefix.

Files and directories are timestamped to keep runs distinct. Review the JSON to capture the generated tweet content and the associated madlib choices.

## Optional: Run Tests

The repository includes unit tests for the prompt builder and generator utilities. Activate the virtual environment and install dev dependencies (if any), then run:

```bash
pytest
```

This ensures your environment can render prompts and handle simulated model responses before invoking Replicate.
