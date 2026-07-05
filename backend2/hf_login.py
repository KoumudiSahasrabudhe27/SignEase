#!/usr/bin/env python3
"""One-time HuggingFace login for the gated IndicTrans2 model."""

from __future__ import annotations

from getpass import getpass
from pathlib import Path

MODEL_PAGE = "https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M"
TOKEN_PAGE = "https://huggingface.co/settings/tokens"


def main() -> None:
    print("IndicTrans2 requires HuggingFace access.\n")
    print("1) Open this page and click Agree and access:")
    print(f"   {MODEL_PAGE}\n")
    print("2) Create a Read token here:")
    print(f"   {TOKEN_PAGE}\n")

    token = getpass("Paste your HuggingFace token (input hidden): ").strip()
    if not token:
        raise SystemExit("No token provided.")

    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)

    env_path = Path(__file__).resolve().parent / ".env"
    existing = []
    if env_path.exists():
        existing = [
            line
            for line in env_path.read_text(encoding="utf-8").splitlines()
            if not line.strip().startswith("HF_TOKEN=")
            and not line.strip().startswith("HUGGING_FACE_HUB_TOKEN=")
        ]
    existing.append(f"HF_TOKEN={token}")
    env_path.write_text("\n".join(existing).rstrip() + "\n", encoding="utf-8")

    print(f"\nToken saved to {env_path}")
    print("Start the server with:")
    print("  python3 translation_api.py")


if __name__ == "__main__":
    main()
