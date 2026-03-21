#!/usr/bin/env python3
"""Sign-Ease backend API with VideoMAE inference + text-to-sign mapping."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote

import numpy as np
import torch
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

SUPPORTED_WORDS: List[str] = [
    "food",
    "hello",
    "help",
    "more",
    "no",
    "please",
    "sad",
    "thank you",
    "water",
    "yes",
]

def _pick_device() -> str:
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_shortest_edge(processor: VideoMAEImageProcessor) -> int:
    size_cfg = processor.size
    if isinstance(size_cfg, dict):
        if "shortest_edge" in size_cfg:
            return int(size_cfg["shortest_edge"])
        if "height" in size_cfg and "width" in size_cfg:
            return int(min(size_cfg["height"], size_cfg["width"]))
    if isinstance(size_cfg, int):
        return int(size_cfg)
    return 224


def _resize_shortest_edge(frame_rgb: np.ndarray, shortest_edge: int) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    if min(h, w) <= 0 or min(h, w) == shortest_edge:
        return frame_rgb
    scale = float(shortest_edge) / float(min(h, w))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)


def _uniform_sample_to_16(frames_rgb: List[np.ndarray]) -> List[np.ndarray]:
    if len(frames_rgb) == 0:
        raise ValueError("No frames available.")
    if len(frames_rgb) >= 16:
        idx = np.linspace(0, len(frames_rgb) - 1, 16).round().astype(int).tolist()
        return [frames_rgb[i] for i in idx]
    return frames_rgb + [frames_rgb[-1]] * (16 - len(frames_rgb))


def _b64_to_rgb_image(b64: str) -> np.ndarray:
    import base64

    if "," in b64:
        b64 = b64.split(",", 1)[1]

    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode base64 image")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _decode_images_from_request(files) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for f in files:
        data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            frames.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return frames


def _find_word_folder(dataset_root: Path, word: str) -> Optional[Path]:
    if not dataset_root.exists():
        return None
    aliases = {
        word.lower(),
        word.lower().replace(" ", "_"),
        word.lower().replace("_", " "),
        word.title(),
        word.title().replace(" ", "_"),
    }
    for p in sorted(dataset_root.iterdir()):
        if p.is_dir() and (p.name in aliases or p.name.lower() in {a.lower() for a in aliases}):
            return p
    return None


def _first_video_for_word(dataset_root: Path, word: str) -> Optional[Path]:
    folder = _find_word_folder(dataset_root, word)
    if folder is None:
        return None
    vids = sorted(
        [
            p
            for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in {".mp4", ".mov"}
        ]
    )
    return vids[0] if vids else None


def create_app() -> Flask:
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir.parent / "SignEase_Final_Model"
    if not model_dir.exists():
        model_dir = base_dir / "SignEase_Final_Model"

    dataset_root = base_dir.parent / "SignEase_dataset"
    if not dataset_root.exists():
        dataset_root = base_dir / "SignEase_dataset"

    if not model_dir.exists():
        raise FileNotFoundError(f"VideoMAE model folder not found: {model_dir}")

    device = _pick_device()
    processor = VideoMAEImageProcessor.from_pretrained(str(model_dir))
    model = VideoMAEForVideoClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()
    shortest_edge = _get_shortest_edge(processor)
    stream_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=32))

    app = Flask(__name__)
    CORS(
        app,
        resources={r"/*": {"origins": "*"}},
        supports_credentials=True,
        methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    @app.after_request
    def _add_cors_headers(resp):
        # Ensure explicit method/header allowance for strict browsers
        resp.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        resp.headers.setdefault("Access-Control-Allow-Headers", "Content-Type, Authorization")
        return resp

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "device": device,
                "model_dir": str(model_dir),
                "dataset_root": str(dataset_root),
            }
        )

    @app.route("/predict", methods=["POST", "OPTIONS"])
    def predict():
        try:
            if request.method == "OPTIONS":
                return ("", 204)

            frames_rgb: List[np.ndarray] = []
            if request.is_json:
                payload = request.get_json(silent=True) or {}
                b64 = payload.get("image") or payload.get("frame")
                if not b64:
                    return jsonify({"error": "JSON body must include 'image'."}), 400
                frame = _b64_to_rgb_image(str(b64))
                key = request.remote_addr or "local"
                stream_buffers[key].append(frame)
                frames_rgb = list(stream_buffers[key])
            else:
                files = request.files.getlist("files")
                if files:
                    frames_rgb = _decode_images_from_request(files)

            if not frames_rgb:
                return jsonify({"error": "No decodable frames provided."}), 400

            # VideoMAE requires 16-frame clip
            frames_rgb = _uniform_sample_to_16(frames_rgb)
            frames_rgb = [_resize_shortest_edge(fr, shortest_edge) for fr in frames_rgb]
            inputs = processor(frames_rgb, return_tensors="pt", do_resize=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits[0]
                probs = torch.softmax(logits, dim=-1)

            topk = min(5, probs.numel())
            top_probs, top_idx = torch.topk(probs, k=topk)
            pred = int(top_idx[0].item())
            conf = float(top_probs[0].item())
            id2label = model.config.id2label or {}

            top5 = [
                {
                    "class": int(i.item()),
                    "prob": float(p.item()),
                    "label": str(id2label.get(int(i.item()), f"class_{int(i.item())}")),
                }
                for p, i in zip(top_probs, top_idx)
            ]
            text = str(id2label.get(pred, f"class_{pred}"))

            return jsonify(
                {
                    "text": text,
                    "prediction": text,
                    "pred_class": pred,
                    "confidence": conf,
                    "top5": top5,
                }
            )

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/get_sign_video/<word>", methods=["GET", "OPTIONS"])
    def get_sign_video(word: str):
        if request.method == "OPTIONS":
            return ("", 204)
        norm = word.strip().lower()
        vid = _first_video_for_word(dataset_root, norm)
        if vid is None:
            return jsonify({"error": f"No video found for '{word}'", "dataset_root": str(dataset_root)}), 404
        return jsonify(
            {
                "word": norm,
                "file_path": str(vid),
                "video_url": f"/sign_video_file/{quote(norm)}",
            }
        )

    @app.route("/sign_video_file/<word>", methods=["GET"])
    def sign_video_file(word: str):
        norm = word.strip().lower()
        vid = _first_video_for_word(dataset_root, norm)
        if vid is None:
            return jsonify({"error": f"No video found for '{word}'"}), 404
        return send_file(str(vid), mimetype="video/mp4")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8001, debug=True)

