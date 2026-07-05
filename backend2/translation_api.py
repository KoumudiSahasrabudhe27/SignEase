#!/usr/bin/env python3
"""Flask API for IndicTrans2 English → Indic translation (port 8002)."""

from __future__ import annotations

import os
import traceback

from load_env import load_backend_env

load_backend_env()

from flask import Flask, jsonify, request
from flask_cors import CORS

from indictrans_service import SOURCE_LANG, SUPPORTED_TARGET_LANGS, IndicTrans2Service

_service: IndicTrans2Service | None = None


def get_service() -> IndicTrans2Service:
    if _service is None:
        raise RuntimeError("Translation service is not initialized.")
    return _service


def _payload_field(payload: dict, *keys: str, default: str = "") -> str:
    for key in keys:
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})

    @app.after_request
    def _add_cors_headers(resp):
        resp.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type, Authorization")
        return resp

    @app.get("/health")
    def health():
        svc = get_service()
        return jsonify(
            {
                "status": "ok",
                "device": svc.device,
                "model_id": svc.model_id,
                "source_lang": SOURCE_LANG,
                "loaded": svc._loaded,
            }
        )

    @app.route("/translate", methods=["POST", "OPTIONS"])
    def translate():
        if request.method == "OPTIONS":
            return ("", 204)

        payload = request.get_json(silent=True) or {}
        text = payload.get("text", "")
        src_lang = _payload_field(
            payload, "source_lang", "source_language", default=SOURCE_LANG
        )
        tgt_lang = _payload_field(payload, "target_lang", "target_language")

        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "Request body must include non-empty 'text'."}), 400
        if not tgt_lang:
            return jsonify({"error": "Request body must include 'target_lang'."}), 400
        if tgt_lang not in SUPPORTED_TARGET_LANGS:
            return jsonify({"error": f"Unsupported target language: {tgt_lang}"}), 400

        try:
            translation = get_service().translate(text, src_lang, tgt_lang)
            return jsonify({"translation": translation})
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception:
            traceback.print_exc()
            return jsonify({"error": "Translation failed. Please try again."}), 500

    return app


def main() -> None:
    global _service

    print("Loading IndicTrans2 model...")
    try:
        _service = IndicTrans2Service()
        _service.load()
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc

    print("Model loaded successfully.")

    app = create_app()
    port = int(os.environ.get("TRANSLATION_PORT", "8002"))
    print("Translation server running...")
    print(f"Listening on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
