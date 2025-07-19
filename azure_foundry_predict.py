"""Azure AI Foundry Chat Completion Helper

This script sends a prompt to an Azure AI Foundry (Azure OpenAI) chat deployment
and prints the assistant\'s response.  It can be imported as a module
(`predict(prompt: str)`) or executed directly from the command line.

Environment variables (recommended)
-----------------------------------
Set these variables in your shell or a local `.env` file so that the helper can
run without hard-coding secrets:

* `AZURE_OPENAI_ENDPOINT`   – Your Cognitive Services endpoint URL.
* `AZURE_OPENAI_DEPLOYMENT` – Name of the deployed chat model.
* `AZURE_OPENAI_API_KEY`    – The model subscription key.
* `AZURE_OPENAI_API_VERSION` (optional) – Defaults to `2024-12-01-preview`.

Example CLI usage
-----------------
```
$ export AZURE_OPENAI_ENDPOINT="https://<resource>.cognitiveservices.azure.com/"
$ export AZURE_OPENAI_DEPLOYMENT="gpt-4-1-nano-2025-04-14-ft-1af1a5b552...
$ export AZURE_OPENAI_API_KEY="<your-key>"

$ python azure_foundry_predict.py "Metrics: trunk_angle=2.60, nsar=0.1574, theta_d=178.45. Predict fall risk."
```

or programmatically:

```python
from azure_foundry_predict import predict
result = predict("Metrics: trunk_angle=2.603, nsar=0.157, theta_d=178.45. Predict fall risk.")
print(result)
```
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

from dotenv import load_dotenv  # type: ignore
from openai import AzureOpenAI

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

load_dotenv()  # Load variables from an optional local .env file

_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

_REQUIRED_ENV_VARS: List[str] = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_API_KEY",
]


def _ensure_env() -> None:
    """Validate that all required environment variables are set."""
    missing = [var for var in _REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        msg = (
            "Missing required environment variables: " + ", ".join(missing) + "\n"
            "Set them in your shell or .env file before running."
        )
        print(msg, file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Core prediction function
# ---------------------------------------------------------------------------

def predict(prompt: str, *, temperature: float = 0.0, max_tokens: int = 256) -> str:
    """Send *prompt* to the deployed Azure AI Foundry chat model and return the reply."""

    _ensure_env()

    client = AzureOpenAI(
        api_version=_API_VERSION,
        azure_endpoint=_ENDPOINT,
        api_key=_API_KEY,
    )

    system_prompt = (
        "You are a helpful assistant specialised in assessing human fall risk "
        "based on biomechanical metrics such as trunk angle, NSAR, and mid-plumb "
        "angles. Provide a concise risk rating (e.g., Low, Medium, High) and a short "
        "justification. If advisable, suggest preventive actions."
    )

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        model=_DEPLOYMENT,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Query an Azure AI Foundry chat model with a single prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("prompt", help="The user prompt to send to the model.")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    p.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens in the reply.")
    return p.parse_args()


def main() -> None:  # noqa: D401
    args = _parse_args()
    result = predict(args.prompt, temperature=args.temperature, max_tokens=args.max_tokens)
    print(result)


if __name__ == "__main__":
    main() 