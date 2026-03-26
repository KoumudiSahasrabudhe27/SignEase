#!/usr/bin/env python3
"""Sign-Ease backend API with VideoMAE inference + text-to-sign mapping."""

from __future__ import annotations

import os
import threading
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
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

# LIFO stream: only the 16 most recent frames (drops oldest when full).
STREAM_BUFFER_MAXLEN = 16

# ImageNet normalization (matches common ViT / VideoMAE fine-tuning)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Sliding-window vote on last N *inference* outputs (winner takes all)
VOTE_WINDOW = 10
VOTE_MIN_AGREE = 7

# Require top1 - top2 margin on temperature-scaled probs
MARGIN_MIN = 0.08

# Run VideoMAE every N-th client request; lower N updates neutral / null state faster.
INFERENCE_STRIDE = 3

# Softmax temperature T>1 spreads mass → sharper relative separation in UI
SOFTMAX_TEMPERATURE = 2.0

# Low-confidence gating (for demo persistence rules)
NEUTRAL_HARD_MAX = 0.05
NEUTRAL_SOFT_MAX = 0.15
# Demo boost for challenging classes (applied post-softmax, then renormalized)
PROBABILITY_BOOST_FACTOR = 1.2
BOOST_CLASSES = {"yes", "help"}


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


def _center_crop_square_resize(frame_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    """Center square crop (focus on signer), then resize to size×size."""
    h, w = frame_rgb.shape[:2]
    if h <= 0 or w <= 0:
        return frame_rgb
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cropped = frame_rgb[y0 : y0 + side, x0 : x0 + side]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)


def _label_key(label: str) -> str:
    return " ".join(label.strip().lower().split())


def _is_hard_neutral(top_prob: float) -> bool:
    return top_prob < NEUTRAL_HARD_MAX


def _is_soft_uncertain(top_prob: float) -> bool:
    return NEUTRAL_HARD_MAX <= top_prob < NEUTRAL_SOFT_MAX


def _apply_imagenet_processor_norm(processor: VideoMAEImageProcessor) -> None:
    processor.image_mean = list(IMAGENET_MEAN)
    processor.image_std = list(IMAGENET_STD)


def _uniform_sample_to_16(frames_rgb: List[np.ndarray]) -> List[np.ndarray]:
    """Exactly 16 equidistant frames from the buffer (newest bias when buffer is full)."""
    if len(frames_rgb) == 0:
        raise ValueError("No frames available.")
    n = len(frames_rgb)
    if n >= 16:
        idx = np.linspace(0, n - 1, 16).round().astype(int).tolist()
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


def _normalize_word_key(word: str) -> str:
    """Lowercase and remove spaces/underscores, e.g. 'Thank You' -> 'thankyou'."""
    w = word.strip().lower()
    return w.replace(" ", "").replace("_", "")


def _find_word_folder(dataset_root: Path, word: str) -> Optional[Path]:
    if not dataset_root.exists():
        return None
    key = _normalize_word_key(word)
    wl = word.strip().lower()
    aliases = {
        key,
        wl,
        wl.replace(" ", "_"),
        wl.replace(" ", ""),
    }
    for p in sorted(dataset_root.iterdir()):
        if not p.is_dir():
            continue
        pn = p.name.lower()
        pn_compact = pn.replace(" ", "").replace("_", "")
        if pn in aliases or pn_compact == key or pn_compact in aliases:
            return p
    return None


def _first_mov_in_word_folder(folder: Path) -> Optional[Path]:
    """Prefer the first .mov in the word folder (dataset convention), then .mp4."""
    files = [p for p in folder.iterdir() if p.is_file()]
    movs = sorted(p for p in files if p.suffix.lower() == ".mov")
    if movs:
        return movs[0]
    mp4s = sorted(p for p in files if p.suffix.lower() == ".mp4")
    return mp4s[0] if mp4s else None


def _first_video_for_word(dataset_root: Path, word: str) -> Optional[Path]:
    folder = _find_word_folder(dataset_root, word)
    if folder is None:
        return None
    direct = _first_mov_in_word_folder(folder)
    if direct is not None:
        return direct
    vids = sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in {".mp4", ".mov"}
    )
    return vids[0] if vids else None


def _resolve_dataset_root(base_dir: Path) -> Path:
    """Prefer ./SignEase_Project/SignEase_dataset next to backend2."""
    primary = (base_dir / "SignEase_Project" / "SignEase_dataset").resolve()
    if primary.exists():
        return primary
    for p in (base_dir.parent / "SignEase_dataset", base_dir / "SignEase_dataset"):
        if p.exists():
            return p.resolve()
    return primary


def _resolve_video_mae_dir(base_dir: Path) -> Path:
    """
    Find ./SignEase_Final_Model (Hugging Face export with config + weights).

    Search order:
    1) SIGNEASE_VIDEOMAE_DIR environment variable
    2) <project_root>/SignEase_Final_Model  (parent of backend2/)
    3) backend2/SignEase_Final_Model
    4) backend2/SignEase_Project/SignEase_Final_Model  (common local layout)
    5) current working directory ./SignEase_Final_Model
    """
    candidates: List[Path] = []
    env = os.environ.get("SIGNEASE_VIDEOMAE_DIR", "").strip()
    if env:
        candidates.append(Path(env).expanduser().resolve())
    candidates.append((base_dir.parent / "SignEase_Final_Model").resolve())
    candidates.append((base_dir / "SignEase_Final_Model").resolve())
    candidates.append((base_dir / "SignEase_Project" / "SignEase_Final_Model").resolve())
    candidates.append((Path.cwd() / "SignEase_Final_Model").resolve())

    seen = set()
    unique: List[Path] = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique.append(p)
        if p.exists() and p.is_dir():
            return p

    tried = "\n  ".join(str(p) for p in unique)
    raise FileNotFoundError(
        "VideoMAE model folder 'SignEase_Final_Model' not found.\n"
        "Place your fine-tuned export at one of:\n"
        f"  {tried}\n"
        "Or set: export SIGNEASE_VIDEOMAE_DIR=/path/to/SignEase_Final_Model"
    )


def create_app() -> Flask:
    base_dir = Path(__file__).resolve().parent
    model_dir = _resolve_video_mae_dir(base_dir)

    dataset_root = _resolve_dataset_root(base_dir)

    device = _pick_device()
    processor = VideoMAEImageProcessor.from_pretrained(str(model_dir))
    _apply_imagenet_processor_norm(processor)
    model = VideoMAEForVideoClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()
    crop_size = 224
    stream_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=STREAM_BUFFER_MAXLEN))
    vote_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=VOTE_WINDOW))
    latch_by_client: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {"text": "", "confidence": 0.0}
    )
    infer_frame_counters: Dict[str, int] = defaultdict(int)
    last_predict_cache: Dict[str, Dict[str, Any]] = {}
    gpu_infer_lock = threading.Lock()

    def _clear_client_state(k: str) -> None:
        vote_buffers[k].clear()
        stream_buffers[k].clear()
        infer_frame_counters[k] = 0
        latch_by_client[k] = {"text": "", "confidence": 0.0}
        last_predict_cache.pop(k, None)

    def _cache_response(
        key: str,
        *,
        prediction_display: Optional[str],
        latched_conf: float,
        relative_conf: float,
        text: str,
        pred: int,
        conf: float,
        margin: float,
        vote_winner_display: Optional[str],
        vote_support: int,
        top5: List[dict],
        neutral_inference: bool = False,
        skipped: str = "",
    ) -> Dict[str, Any]:
        out_pred = prediction_display.strip() if prediction_display and prediction_display.strip() else None
        neutral = neutral_inference or (out_pred is None)
        disp_conf = latched_conf if out_pred is not None else conf
        body: Dict[str, Any] = {
            "text": out_pred,
            "prediction": out_pred,
            "raw_prediction": text,
            "pred_class": pred,
            "confidence": disp_conf,
            "inference_confidence": relative_conf,
            "inference_confidence_raw": conf,
            "confirmed": bool(out_pred),
            "margin": margin,
            "vote_support": vote_support,
            "vote_winner": vote_winner_display or "",
            "top5": top5,
            "neutral": neutral,
        }
        if skipped:
            body["inference_skipped"] = skipped
        last_predict_cache[key] = body
        return body

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
            key = request.remote_addr or "local"
            is_json_stream = False
            if request.is_json:
                is_json_stream = True
                payload = request.get_json(silent=True) or {}
                if payload.get("reset_ui_only"):
                    _clear_client_state(key)
                    body = _cache_response(
                        key,
                        prediction_display=None,
                        latched_conf=0.0,
                        relative_conf=0.0,
                        text="",
                        pred=-1,
                        conf=0.0,
                        margin=0.0,
                        vote_winner_display=None,
                        vote_support=0,
                        top5=[],
                        neutral_inference=True,
                    )
                    return jsonify(body)
                if payload.get("reset_session"):
                    _clear_client_state(key)
                b64 = payload.get("image") or payload.get("frame")
                if not b64:
                    return jsonify({"error": "JSON body must include 'image'."}), 400
                frame = _b64_to_rgb_image(str(b64))
                stream_buffers[key].append(frame)
                frames_rgb = list(stream_buffers[key])
            else:
                files = request.files.getlist("files")
                if files:
                    frames_rgb = _decode_images_from_request(files)

            if not frames_rgb:
                return jsonify({"error": "No decodable frames provided."}), 400

            def _return_cached(skipped: str) -> Any:
                if key in last_predict_cache:
                    out = dict(last_predict_cache[key])
                    out["inference_skipped"] = skipped
                    return jsonify(out)
                lat = latch_by_client[key]
                latched = str(lat.get("text") or "").strip()
                latched_conf = float(lat.get("confidence") or 0.0)
                if latched:
                    return jsonify(
                        {
                            "text": latched,
                            "prediction": latched,
                            "raw_prediction": "",
                            "pred_class": -1,
                            "confidence": latched_conf,
                            "inference_confidence": latched_conf,
                            "margin": 0.0,
                            "vote_support": 0,
                            "vote_winner": "",
                            "top5": [],
                            "neutral": False,
                            "inference_skipped": skipped,
                        }
                    )
                return jsonify(
                    {
                        "text": None,
                        "prediction": None,
                        "raw_prediction": "",
                        "pred_class": -1,
                        "confidence": 0.0,
                        "inference_confidence": 0.0,
                        "margin": 0.0,
                        "vote_support": 0,
                        "vote_winner": "",
                        "top5": [],
                        "neutral": True,
                        "inference_skipped": skipped,
                    }
                )

            # Throttle: run inference on ticks 0, STRIDE, 2*STRIDE, ... (JSON stream only).
            if is_json_stream:
                tick = infer_frame_counters[key]
                infer_frame_counters[key] = tick + 1
                if tick % INFERENCE_STRIDE != 0:
                    return _return_cached("stride")

            # If another request is still in forward(), return immediately (capture never blocks on GPU).
            if not gpu_infer_lock.acquire(blocking=False):
                return _return_cached("busy")

            try:
                # 16 equidistant samples from the 16-slot LIFO buffer, then center-crop 224².
                frames_proc = _uniform_sample_to_16(frames_rgb)
                frames_proc = [_center_crop_square_resize(fr, crop_size) for fr in frames_proc]
                inputs = processor(frames_proc, return_tensors="pt", do_resize=False)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    logits = model(**inputs).logits[0]
                    scaled = logits / float(SOFTMAX_TEMPERATURE)
                    probs = torch.softmax(scaled, dim=-1)
                    # 1.2x probability boost for HELP/YES classes, then renormalize.
                    id2label_for_boost = model.config.id2label or {}
                    for idx in range(int(probs.shape[0])):
                        lk = _label_key(str(id2label_for_boost.get(idx, f"class_{idx}")))
                        if lk in BOOST_CLASSES:
                            probs[idx] = probs[idx] * float(PROBABILITY_BOOST_FACTOR)
                    probs = probs / torch.clamp(probs.sum(), min=1e-9)

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
                text = str(id2label.get(pred, f"class_{pred}")).strip()
                margin = float(top_probs[0].item() - top_probs[1].item()) if topk >= 2 else float(top_probs[0].item())
                top2 = float(top_probs[1].item()) if topk >= 2 else 0.0
                # Relative dominance score: 25% vs 5% => 83.3%
                relative_conf = conf / max(1e-9, (conf + top2))

                lat = latch_by_client[key]
                if _is_hard_neutral(conf):
                    # Keep last confirmed sign on screen unless reset_session is clicked.
                    latched_s = str(lat.get("text") or "").strip()
                    latched_conf_v = float(lat.get("confidence") or 0.0)
                    body = _cache_response(
                        key,
                        prediction_display=latched_s if latched_s else None,
                        latched_conf=latched_conf_v,
                        relative_conf=relative_conf,
                        text=text,
                        pred=pred,
                        conf=conf,
                        margin=margin,
                        vote_winner_display=None,
                        vote_support=0,
                        top5=top5,
                        neutral_inference=True,
                    )
                    return jsonify(body)

                if _is_soft_uncertain(conf):
                    latched_s = str(lat.get("text") or "").strip()
                    latched_conf_v = float(lat.get("confidence") or 0.0)
                    body = _cache_response(
                        key,
                        prediction_display=latched_s if latched_s else None,
                        latched_conf=latched_conf_v,
                        relative_conf=relative_conf,
                        text=text,
                        pred=pred,
                        conf=conf,
                        margin=margin,
                        vote_winner_display=None,
                        vote_support=0,
                        top5=top5,
                        neutral_inference=not bool(latched_s),
                    )
                    return jsonify(body)

                if text:
                    vote_buffers[key].append(text)
                window = list(vote_buffers[key])
                vote_winner_display: Optional[str] = None
                vote_support = 0
                if len(window) >= VOTE_MIN_AGREE:
                    keys = [_label_key(x) for x in window]
                    for k_common, cnt_v in Counter(keys).most_common():
                        if cnt_v >= VOTE_MIN_AGREE:
                            for x in reversed(window):
                                if _label_key(x) == k_common:
                                    vote_winner_display = x
                                    vote_support = cnt_v
                                    break
                            break

                if vote_winner_display and margin >= MARGIN_MIN:
                    lat["text"] = vote_winner_display
                    lat["confidence"] = conf
                    stream_buffers[key].clear()

                latched = str(lat.get("text") or "")
                latched_conf = float(lat.get("confidence") or 0.0)

                body = _cache_response(
                    key,
                    prediction_display=latched if latched else None,
                    latched_conf=latched_conf,
                    relative_conf=relative_conf,
                    text=text,
                    pred=pred,
                    conf=conf,
                    margin=margin,
                    vote_winner_display=vote_winner_display,
                    vote_support=vote_support,
                    top5=top5,
                    neutral_inference=False,
                )
                return jsonify(body)
            finally:
                gpu_infer_lock.release()

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def _mimetype_for_video(path: Path) -> str:
        return "video/quicktime" if path.suffix.lower() == ".mov" else "video/mp4"

    @app.route("/get_sign_video/<word>", methods=["GET", "OPTIONS"])
    def get_sign_video(word: str):
        """Serve first .mov in ./SignEase_Project/SignEase_dataset/<word-folder>/ (else .mp4)."""
        if request.method == "OPTIONS":
            return ("", 204)
        vid = _first_video_for_word(dataset_root, word)
        if vid is None:
            return jsonify({"error": f"No video found for '{word}'", "dataset_root": str(dataset_root)}), 404
        folder = vid.parent
        return send_from_directory(
            str(folder.resolve()),
            vid.name,
            mimetype=_mimetype_for_video(vid),
        )

    @app.route("/sign_video_file/<word>", methods=["GET", "OPTIONS"])
    def sign_video_file(word: str):
        if request.method == "OPTIONS":
            return ("", 204)
        vid = _first_video_for_word(dataset_root, word)
        if vid is None:
            return jsonify({"error": f"No video found for '{word}'"}), 404
        folder = vid.parent
        return send_from_directory(
            str(folder.resolve()),
            vid.name,
            mimetype=_mimetype_for_video(vid),
        )

    return app


if __name__ == "__main__":
    app = create_app()
    # Default 8001: macOS often reserves 5000 for AirPlay Receiver.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8001")), debug=True)

