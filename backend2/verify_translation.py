#!/usr/bin/env python3
"""Verify IndicTrans2 translations (run after model access is configured)."""

from indictrans_service import IndicTrans2Service

TESTS = [
    ("hello", "hin_Deva"),
    ("good morning", "mar_Deva"),
    ("how are you", "tam_Taml"),
]


def main() -> None:
    print("Loading IndicTrans2 model...")
    service = IndicTrans2Service()
    service.load()
    print("Model loaded successfully.\n")

    for text, target in TESTS:
        result = service.translate(text, "eng_Latn", target)
        print(f"{text!r} -> {target}: {result!r}")

    print("\nAll verification tests completed.")


if __name__ == "__main__":
    main()
