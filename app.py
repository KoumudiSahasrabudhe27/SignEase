#!/usr/bin/env python3
"""
Sign-Ease Hybrid Spatiotemporal Inference App (VideoMAE + Streamlit).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
import streamlit as st
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor


SUPPORTED_WORDS: List[str] = [
    "Food",
    "Hello",
    "Help",
    "More",
    "No",
    "Please",
    "Sad",
    "Thank You",
    "Water",
    "Yes",
]

MODEL_DIR = Path(__file__).resolve().parent / "SignEase_Final_Model"


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
    if min(h, w) == shortest_edge:
        return frame_rgb

    scale = float(shortest_edge) / float(min(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _sample_exactly_16_frames(video_path: Path) -> List[np.ndarray]:
    """Uniformly sample 16 RGB frames to match VideoMAE fine-tuning (same clip length as the API)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames_bgr: List[np.ndarray] = []
    if total_frames <= 0:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames_bgr.append(fr)
    else:
        idx = np.linspace(0, total_frames - 1, 16).round().astype(int).tolist()
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if not ok or fr is None:
                ok2, fr2 = cap.read()
                if not ok2 or fr2 is None:
                    break
                fr = fr2
            frames_bgr.append(fr)

    cap.release()

    if not frames_bgr:
        raise RuntimeError("No readable frames were found in the uploaded video.")

    if len(frames_bgr) >= 16:
        idx = np.linspace(0, len(frames_bgr) - 1, 16).round().astype(int).tolist()
        frames_bgr = [frames_bgr[i] for i in idx]
    else:
        frames_bgr += [frames_bgr[-1]] * (16 - len(frames_bgr))

    frames_rgb = [cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) for fr in frames_bgr]
    return frames_rgb


@st.cache_resource(show_spinner=True)
def load_model_and_processor():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model folder not found: {MODEL_DIR}")
    processor = VideoMAEImageProcessor.from_pretrained(str(MODEL_DIR))
    model = VideoMAEForVideoClassification.from_pretrained(str(MODEL_DIR))
    model.eval()
    return processor, model


def main():
    st.set_page_config(page_title="Sign-Ease VideoMAE", page_icon="🤟", layout="centered")
    st.title("Sign-Ease: Hybrid Spatiotemporal Sign Recognition")
    st.caption("VideoMAE-based inference for robust word-level sign recognition")

    with st.sidebar:
        st.header("Supported Words (10)")
        for w in SUPPORTED_WORDS:
            st.write(f"- {w}")

    st.markdown(
        """
        ### Technical Note
        This model uses **Temporal Tubelet Embedding** and **Masked Video Modeling** (VideoMAE)
        to capture the motion trajectory of hands across time. This spatiotemporal modeling is
        significantly stronger for dynamic gestures than a standard frame-only Vision Transformer.
        """
    )

    uploaded = st.file_uploader("Upload a sign video (.mp4 or .mov)", type=["mp4", "mov"])
    if uploaded is None:
        return

    processor, model = load_model_and_processor()
    shortest_edge = _get_shortest_edge(processor)

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = Path(tmp.name)

    try:
        with st.spinner("Processing video and running inference..."):
            frames = _sample_exactly_16_frames(tmp_path)
            frames = [_resize_shortest_edge(fr, shortest_edge=shortest_edge) for fr in frames]

            # We already handle shortest_edge resize above; disable resize in processor.
            inputs = processor(
                frames,
                return_tensors="pt",
                do_resize=False,
            )

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
                pred_idx = int(torch.argmax(probs).item())
                confidence = float(probs[pred_idx].item())

            if model.config.id2label and pred_idx in model.config.id2label:
                pred_label = model.config.id2label[pred_idx]
            elif pred_idx < len(SUPPORTED_WORDS):
                pred_label = SUPPORTED_WORDS[pred_idx]
            else:
                pred_label = f"class_{pred_idx}"

        st.success(f"Prediction: {pred_label}  |  Confidence: {confidence * 100:.2f}%")
        st.markdown(
            f"<div style='font-size: 1.5rem; font-weight: 700; margin-top: 0.5rem;'>✅ {pred_label}</div>",
            unsafe_allow_html=True,
        )

    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()

