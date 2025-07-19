"""Azure ML Fall-Detection Inference Client

This utility posts a simple JSON payload containing fall-detection metrics to an
Azure Machine Learning managed-online endpoint (or any compatible REST
endpoint) and prints the scored result.

The payload schema expected by the endpoint is a flat JSON object:

    {
        "trunk_angle": <float>,
        "nsar": <float>,
        "theta_u": <float>,
        "theta_d": <float>
    }

Both the endpoint URL and the authentication key can be supplied via
command-line arguments or environment variables. The latter are convenient if
you prefer not to expose secrets in your shell history:

    AML_ENDPOINT_URL  – The full scoring URL (e.g. "https://.../score")
    AML_ENDPOINT_KEY  – The primary/secondary key for the endpoint

Example
-------
$ python aml_fall_detection_client.py \
    --endpoint "https://my-workspace.eastus.inference.ml.azure.com/score" \
    --api-key "<your-key>" \
    --trunk-angle 35.2 --nsar 0.22 --theta-u 50.8 --theta-d 75.6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import requests

__all__ = [
    "send_metrics",
]


def _parse_args() -> argparse.Namespace:  # pragma: no cover
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Send fall-detection metrics to an Azure ML REST endpoint and print the response.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.getenv("AML_ENDPOINT_URL"),
        help="Full URL of the Azure ML endpoint (can be set via AML_ENDPOINT_URL).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("AML_ENDPOINT_KEY"),
        help="Endpoint API key for the *Authorization* header (can be set via AML_ENDPOINT_KEY).",
    )
    parser.add_argument("--trunk-angle", type=float, required=True, help="Smoothed trunk angle in degrees.")
    parser.add_argument("--nsar", type=float, required=True, help="Normalized shoulder-to-ankle ratio.")
    parser.add_argument("--theta-u", type=float, required=True, help="Upper plumb angle θ₍ᵤ₎ in degrees.")
    parser.add_argument("--theta-d", type=float, required=True, help="Lower plumb angle θ₍d₎ in degrees.")

    args = parser.parse_args()

    if not args.endpoint:
        parser.error("--endpoint not provided and AML_ENDPOINT_URL not set.")
    if not args.api_key:
        parser.error("--api-key not provided and AML_ENDPOINT_KEY not set.")

    return args


def _build_payload(args: argparse.Namespace) -> Dict[str, float]:
    """Construct the JSON payload expected by the model endpoint."""
    return {
        "trunk_angle": args.trunk_angle,
        "nsar": args.nsar,
        "theta_u": args.theta_u,
        "theta_d": args.theta_d,
    }


def send_metrics(
    endpoint_url: str,
    api_key: str,
    metrics: Dict[str, float],
    timeout: Optional[float] = 10.0,
) -> Any:
    """Send *metrics* JSON to *endpoint_url* and return the parsed response.

    Parameters
    ----------
    endpoint_url : str
        Fully-qualified scoring URL.
    api_key : str
        Primary/secondary access key for the endpoint.
    metrics : Dict[str, float]
        Dictionary matching the expected payload schema.
    timeout : float | None, optional
        HTTP timeout in seconds for the request. Defaults to 10 s.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    response = requests.post(endpoint_url, headers=headers, data=json.dumps(metrics), timeout=timeout)
    response.raise_for_status()  # Raise an exception for non-200 responses
    try:
        return response.json()
    except ValueError:  # response not JSON
        return response.text


if __name__ == "__main__":  # pragma: no cover
    cli_args = _parse_args()
    payload = _build_payload(cli_args)

    print("Sending payload:")
    print(json.dumps(payload, indent=2))

    try:
        result = send_metrics(cli_args.endpoint, cli_args.api_key, payload)
    except requests.RequestException as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\nReceived response:")
    print(json.dumps(result, indent=2) if isinstance(result, dict) else result) 