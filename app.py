#!/usr/bin/env python3
"""
SignEase ASL Recognition Demo — VideoMAE spatiotemporal inference (Streamlit).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import base64
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

ROOT = Path(__file__).resolve().parent
LABEL_MAP_PATH = ROOT / "label_map.json"
FRAME_SIZE = 224
NUM_FRAMES = 16
SHARPEN_TEMPERATURE = 0.5
LOCK_THRESHOLD = 85.0


def _resolve_model_dir() -> Path:
    candidates = [
        ROOT / "SignEase_final_final" / "SignEase_Final_Model",
        ROOT / "backend2" / "SignEase_final_final" / "SignEase_Final_Model",
    ]
    for p in candidates:
        if p.is_dir() and (p / "config.json").is_file():
            weights = list(p.glob("model.safetensors")) + list(p.glob("pytorch_model.bin"))
            if weights:
                return p
    raise FileNotFoundError(
        "Could not find VideoMAE weights under SignEase_final_final/SignEase_Final_Model. "
        "Place the exported model next to the repo root or under backend2/."
    )


@st.cache_data(show_spinner=False)
def load_label_map(path_str: str) -> Dict[int, str]:
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"label_map.json not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): str(v) for k, v in raw.items()}


def _slug_to_display(slug: str) -> str:
    if slug == "thankyou":
        return "Thank You"
    return slug.replace("_", " ").capitalize()


def _sample_uniform_frames_bgr(video_path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames_bgr: List[np.ndarray] = []

    if total <= 0:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames_bgr.append(fr)
    else:
        indices = np.linspace(0, total - 1, NUM_FRAMES).round().astype(int).tolist()
        for i in indices:
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

    if len(frames_bgr) >= NUM_FRAMES:
        pick = np.linspace(0, len(frames_bgr) - 1, NUM_FRAMES).round().astype(int).tolist()
        frames_bgr = [frames_bgr[i] for i in pick]
    else:
        frames_bgr = frames_bgr + [frames_bgr[-1]] * (NUM_FRAMES - len(frames_bgr))

    return frames_bgr


def _resize_frames_rgb(frames_bgr: List[np.ndarray], size: int) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for fr in frames_bgr:
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        out.append(cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR))
    return out


def _resolve_dataset_dir() -> Optional[Path]:
    candidates = [
        ROOT / "SignEase_final_final" / "SignEase_dataset",
        ROOT / "SignEase_dataset",
        ROOT / "SignEase_Project" / "SignEase_dataset",
        ROOT / "backend2" / "SignEase_final_final" / "SignEase_dataset",
        ROOT / "backend2" / "SignEase_dataset",
        ROOT / "backend2" / "SignEase_Project" / "SignEase_dataset",
    ]
    for p in candidates:
        if p.is_dir():
            return p.resolve()
    return None


def _normalize_word_key(word: str) -> str:
    return word.strip().lower().replace(" ", "").replace("_", "")


def get_reference_video(sign_name: str) -> Optional[str]:
    """Direct local loader: first video file in SignEase_dataset/[sign_name]/."""
    dataset_dir = _resolve_dataset_dir()
    if dataset_dir is None:
        return None

    target_key = _normalize_word_key(sign_name)
    target_dir = None
    for sub in sorted(dataset_dir.iterdir()):
        if sub.is_dir() and _normalize_word_key(sub.name) == target_key:
            target_dir = str(sub.resolve())
            break
    if target_dir is None:
        return None

    video_exts = {".mp4", ".mov", ".webm", ".m4v"}
    for name in sorted(os.listdir(target_dir)):
        path = os.path.join(target_dir, name)
        if os.path.isfile(path) and Path(path).suffix.lower() in video_exts:
            return path
    return None


def render_video_base64(video_path: str) -> None:
    raw = Path(video_path).read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    ext = Path(video_path).suffix.lower().lstrip(".") or "mp4"
    mime = "video/mp4" if ext in {"mp4", "m4v"} else f"video/{ext}"
    html = f"""
    <div style="display:flex; justify-content:center; align-items:center; width:100%;">
      <video controls autoplay loop muted playsinline style="width:420px; max-width:100%; border-radius:12px;">
        <source src="data:{mime};base64,{b64}" type="{mime}" />
      </video>
    </div>
    """
    components.html(html, height=340)


def _init_prediction_hold_state() -> None:
    st.session_state.setdefault("last_valid_prediction", None)
    st.session_state.setdefault("last_valid_confidence", 0.0)
    st.session_state.setdefault("last_valid_time", 0.0)


def _apply_peak_hold(pred_label: str, confidence_pct: float) -> tuple[str, float, bool]:
    """
    Peak-Hold rules:
    - If current conf <50 and previous conf >75 within 3s, keep previous.
    - Only switch to a different sign when new sign conf >60.
    """
    _init_prediction_hold_state()
    now = time.time()

    prev_label = st.session_state["last_valid_prediction"]
    prev_conf = float(st.session_state["last_valid_confidence"])
    prev_time = float(st.session_state["last_valid_time"])

    if prev_label and confidence_pct < 70.0 and prev_conf > 75.0 and (now - prev_time) <= 3.0:
        return str(prev_label), prev_conf, True

    if prev_label and confidence_pct < 70.0:
        return str(prev_label), prev_conf, True

    st.session_state["last_valid_prediction"] = pred_label
    st.session_state["last_valid_confidence"] = confidence_pct
    st.session_state["last_valid_time"] = now
    return pred_label, confidence_pct, False


@st.cache_resource(show_spinner=True)
def load_model_and_processor(model_dir_str: str):
    model_dir = Path(model_dir_str)
    processor = VideoMAEImageProcessor.from_pretrained(str(model_dir))
    model = VideoMAEForVideoClassification.from_pretrained(str(model_dir))
    model.eval()
    return processor, model


def _inject_dark_theme_css() -> None:
    st.markdown(
        """
<style>
  .stApp {
    background: linear-gradient(165deg, #0b0f14 0%, #121a22 45%, #0e141c 100%);
    color: #e8eef5;
  }
  section[data-testid="stSidebar"] {
    background: #0e131a;
    border-right: 1px solid rgba(255,255,255,0.06);
  }
  div[data-testid="stVerticalBlock"] > div:has(> div.result-card-outer) {
    margin-top: 0.5rem;
  }
  .result-card-outer {
    border-radius: 16px;
    padding: 1px;
    background: linear-gradient(135deg, rgba(56,189,248,0.35), rgba(16,185,129,0.25));
    box-shadow: 0 12px 40px rgba(0,0,0,0.45);
  }
  .result-card-inner {
    border-radius: 15px;
    background: rgba(15, 23, 34, 0.92);
    padding: 1.75rem 1.5rem;
    border: 1px solid rgba(255,255,255,0.06);
  }
  .result-label {
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(148, 163, 184, 0.95);
    margin-bottom: 0.35rem;
  }
  .result-sign {
    font-size: clamp(2rem, 5vw, 3.25rem);
    font-weight: 800;
    line-height: 1.1;
    color: #4ade80;
    text-shadow: 0 0 28px rgba(74, 222, 128, 0.25);
    margin: 0 0 0.75rem 0;
  }
  .result-confidence {
    font-size: 1.35rem;
    font-weight: 700;
    color: #e2e8f0;
  }
  .muted { color: rgba(148, 163, 184, 0.9); font-size: 0.95rem; }
  h1, h2, h3 { color: #f1f5f9 !important; }
</style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="SignEase ASL Demo",
        page_icon="🤟",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_dark_theme_css()

    try:
        model_dir = _resolve_model_dir()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info(
            "Expected model folders: "
            "`SignEase_final_final/SignEase_Final_Model` or "
            "`backend2/SignEase_final_final/SignEase_Final_Model`."
        )
        return

    with st.sidebar:
        st.markdown("### SignEase")
        st.markdown(
            """
**Spatio-temporal transformer**

This demo runs a **VideoMAE** model: a masked video modeling architecture that learns motion and hand trajectories across time—not from isolated frames alone.

**Training data**

Fine-tuned on a **hybrid dataset** combining **SignEase** studio-captured samples with **WLASL** real-world clips for broader coverage and robustness.

**Reported accuracy**

Validation accuracy on the combined setup: **88.1%** (10-class word recognition for this checkpoint).
            """
        )
        st.divider()
        st.caption(f"Model: `{model_dir}`")

    st.title("SignEase ASL Recognition")
    st.caption("Upload a short clip — we sample 16 uniform frames, resize to 224×224, normalize, and classify with VideoMAE.")
    st.markdown("---")

    try:
        label_map = load_label_map(str(LABEL_MAP_PATH))
    except Exception as exc:
        st.error(f"Failed to load label map: {exc}")
        return

    try:
        processor, model = load_model_and_processor(str(model_dir))
    except Exception as exc:
        st.error(f"Failed to load VideoMAE model/processor: {exc}")
        return

    st.session_state.setdefault("is_locked", False)
    st.session_state.setdefault("final_result", "")

    left, middle, right = st.columns([1, 2, 1])
    with middle:
        st.subheader("Sign-to-Text")
        if st.session_state.is_locked:
            final_text = _slug_to_display(str(st.session_state.final_result or ""))
            st.markdown(
                f"""
<div class="result-card-outer">
  <div class="result-card-inner" style="border:2px solid #22c55e; text-align:center;">
    <div class="result-label">SIGN DETECTED</div>
    <p class="result-sign">{final_text}</p>
  </div>
</div>
                """,
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                if st.button("START NEW RECOGNITION", use_container_width=True):
                    st.session_state.is_locked = False
                    st.session_state.final_result = ""
                    st.rerun()
        else:
            uploaded = st.file_uploader("Upload a sign video (.mp4)", type=["mp4"])
            if uploaded is None:
                st.info("Choose an `.mp4` file to run inference.")
                uploaded = None
            if uploaded is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = Path(tmp.name)

                try:
                    with st.spinner("Extracting frames and running VideoMAE…"):
                        frames_bgr = _sample_uniform_frames_bgr(tmp_path)
                        frames_rgb = _resize_frames_rgb(frames_bgr, FRAME_SIZE)

                        inputs = processor(frames_rgb, return_tensors="pt", do_resize=False)

                        with torch.no_grad():
                            outputs = model(**inputs)
                            sharpened_probs = torch.softmax(outputs.logits / SHARPEN_TEMPERATURE, dim=-1)[0]
                            pred_idx = int(torch.argmax(sharpened_probs).item())
                            confidence = float(sharpened_probs[pred_idx].item())

                        sign_slug = label_map.get(pred_idx)
                        if sign_slug is None:
                            st.error(f"Model class index {pred_idx} is outside label_map.json (0–9).")
                            return

                        display_word = _slug_to_display(sign_slug)

                    pct = confidence * 100.0
                    stable_word, stable_pct, held = _apply_peak_hold(display_word, pct)
                    if stable_pct >= LOCK_THRESHOLD:
                        st.session_state.is_locked = True
                        st.session_state.final_result = display_word
                        st.rerun()

                    st.markdown(
                        f"""
<div class="result-card-outer">
  <div class="result-card-inner">
    <div class="result-label">Result</div>
    <p class="result-sign">{stable_word}</p>
    <div class="result-confidence">{stable_pct:.1f}% <span class="muted">confidence</span></div>
  </div>
</div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if held:
                        st.caption("Peak-Hold active: retaining recent high-confidence prediction.")
                except Exception as exc:
                    st.error(f"Unable to process video inference: {exc}")
                finally:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except OSError:
                        pass

    st.markdown("---")
    left2, middle2, right2 = st.columns([1, 2, 1])
    with middle2:
        st.subheader("Text-to-Sign Reference")
        st.caption("Type a word from the 10-class vocabulary to load local reference media from SignEase_dataset.")

        dataset_dir = _resolve_dataset_dir()
        if dataset_dir is None:
            st.error(
                "SignEase_dataset folder not found. Expected one of: "
                "`SignEase_final_final/SignEase_dataset/`, "
                "`SignEase_dataset/`, `SignEase_Project/SignEase_dataset/`, "
                "`backend2/SignEase_final_final/SignEase_dataset/`, "
                "`backend2/SignEase_dataset/`, or `backend2/SignEase_Project/SignEase_dataset/`."
            )
        else:
            st.caption(f"Dataset path: `{dataset_dir}`")
            selected_sign = st.text_input(
                "Search word",
                value="hello",
                placeholder="food, hello, help, more, no, please, sad, thank you, water, yes",
            )
            if selected_sign.strip():
                video_path = get_reference_video(selected_sign)
                if video_path is None:
                    st.error(f"No reference video found for {selected_sign}")
                elif not os.path.exists(video_path):
                    st.error(f"No reference video found for {selected_sign}")
                else:
                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c2:
                        render_video_base64(video_path)


if __name__ == "__main__":
    main()
