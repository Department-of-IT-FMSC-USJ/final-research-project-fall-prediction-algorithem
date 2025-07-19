# telegram_sender.py
"""Send a Telegram message to a user mapped by phone number.

Telegram bots send messages to *chat IDs* – not phone numbers – but in many
projects it's convenient to address users by phone.  This script maintains a
simple mapping from phone numbers (E.164 format) to known chat IDs and delegates
the actual sending to the Telegram Bot API.

Usage examples
--------------
1) Direct chat-id (recommended):

    TELEGRAM_BOT_TOKEN=<token> python telegram_sender.py \
        --chat-id 123456789 --message "Hello from Flask"

2) Via phone-number mapping (requires entry in *phone_chat_map.json*):

    TELEGRAM_BOT_TOKEN=<token> python telegram_sender.py \
        --phone +94123456789 --message "Alert: Fall detected"

Add new mappings by editing *phone_chat_map.json* (see template created on first
run).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import requests  # type: ignore
from dotenv import load_dotenv  # type: ignore

# Load variables from a local .env file if it exists (no error if absent)
load_dotenv()

_MAPPING_FILE = Path("phone_chat_map.json")


def _load_mapping() -> Dict[str, int]:
    if _mapping_file_exists():
        try:
            data = json.loads(_MAPPING_FILE.read_text())
            return {str(k): int(v) for k, v in data.items()}
        except Exception:
            print("[WARN] Could not parse mapping file – starting empty.")
    return {}


def _mapping_file_exists() -> bool:
    return _MAPPING_FILE.exists() and _MAPPING_FILE.is_file()


def _ensure_mapping_template() -> None:
    """Create template mapping file if absent, with instructions."""
    if _mapping_file_exists():
        return
    template = {
        "+1234567890": 123456789,  # replace with your chat_id
    }
    _MAPPING_FILE.write_text(
        json.dumps(template, indent=2)
        + "\n\n"  # newline so editors don't complain
        + "# Edit this file to map phone numbers to Telegram chat IDs.\n"
        + "# WARNING: chat IDs are integers returned by getUpdates / sendMessage.\n"
    )
    print(f"Created template mapping file at {_MAPPING_FILE.resolve()}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Send a Telegram message via bot token.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--message", required=True, help="Text message to send.")
    p.add_argument("--chat-id", type=int, help="Target Telegram chat_id. Can also be set via TELEGRAM_CHAT_ID env var.")
    p.add_argument("--phone", help="Recipient phone number in E.164 format (requires mapping). Can also be set via TELEGRAM_PHONE env var.")
    p.add_argument(
        "--token",
        help="Telegram bot token. If omitted, fall back to TELEGRAM_BOT_TOKEN env variable.",
    )
    return p.parse_args()


def _send(token: str, chat_id: int, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, json={"chat_id": chat_id, "text": text})
    if resp.ok and resp.json().get("ok"):
        print("[OK] Message sent.")
    else:
        print(f"[ERROR] Telegram API error: {resp.text}")
        resp.raise_for_status()


# Helper ---------------------------------------------------------------
def _safe_int(s: str | None) -> int | None:
    """Convert *s* to int if possible, else None."""
    if s is None:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def main() -> None:  # noqa: D401
    args = _parse_args()

    token = args.token or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Error: Telegram bot token not provided (use --token or TELEGRAM_BOT_TOKEN env var).")
        sys.exit(1)

    chat_id: int | None = args.chat_id or _safe_int(os.getenv("TELEGRAM_CHAT_ID"))

    phone = args.phone or os.getenv("TELEGRAM_PHONE")

    if chat_id is None and phone:
        _ensure_mapping_template()
        mapping = _load_mapping()
        chat_id = mapping.get(phone)
        if chat_id is None:
            print(
                f"Phone {phone} not found in mapping file {_MAPPING_FILE}.\n"
                "Add the correct chat_id and try again or set TELEGRAM_CHAT_ID env var."
            )
            sys.exit(1)
    if chat_id is None:
        print("Error: Must provide --chat-id/TELEGRAM_CHAT_ID or --phone/TELEGRAM_PHONE.")
        sys.exit(1)

    _send(token, chat_id, args.message)


if __name__ == "__main__":
    main() 