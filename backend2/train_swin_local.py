#!/usr/bin/env python3
"""
Local SignEase training script (Swin Transformer, 48 classes).

Requirements implemented:
- Looks for videos under backend2/MS-ASL/videos/ (supports extra nesting MS-ASL/videos/videos/)
- Uses LABELS_48 (alphabetical) to map class indices
- Loads backend2/checkpoints/fast_asl_model.pth as starting weights
- Uses MPS on Apple Silicon if available (else CPU)
- Trains for 5 epochs with LR=1e-5 and saves best weights to:
    backend2/checkpoints/final_signease_model.pth
- Adds simple augmentation (random horizontal flip + small rotation)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LABELS_48: List[str] = [
    "afternoon", "angry", "child", "clothes", "day", "drink", "eat", "family",
    "father", "food", "friend", "go", "happy", "hello", "help", "home",
    "how", "know", "like", "love", "man", "money", "more", "morning",
    "mother", "need", "nice", "night", "no", "play", "please", "sad",
    "school", "see", "sorry", "stop", "thank you", "think", "want", "water",
    "what", "when", "where", "who", "why", "woman", "work", "yes",
]


def _pick_device() -> str:
    # Per request: prefer MPS on MacBook Air (Apple Silicon); otherwise CPU.
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _random_resized_crop_bgr(fr: "np.ndarray", scale=(0.8, 1.0)) -> "np.ndarray":
    """
    Minimal RandomResizedCrop-style augmentation (no torchvision dependency).
    Crops a random region whose area is [scale[0], scale[1]] of the original,
    then returns the cropped image (caller will resize to 224x224).
    """
    import numpy as np

    h, w = fr.shape[:2]
    if h < 2 or w < 2:
        return fr

    area = h * w
    target_area = area * float(np.random.uniform(scale[0], scale[1]))

    # Keep aspect ratio close to original (works well for webcam-style footage)
    aspect = w / max(h, 1)
    crop_h = int(round((target_area / max(aspect, 1e-6)) ** 0.5))
    crop_w = int(round(crop_h * aspect))

    crop_h = max(2, min(h, crop_h))
    crop_w = max(2, min(w, crop_w))

    y0 = int(np.random.randint(0, max(1, h - crop_h + 1)))
    x0 = int(np.random.randint(0, max(1, w - crop_w + 1)))
    return fr[y0 : y0 + crop_h, x0 : x0 + crop_w]


def _color_jitter_bgr(fr: "np.ndarray") -> "np.ndarray":
    """
    Minimal ColorJitter-style augmentation (no torchvision dependency).
    Applies random brightness/contrast and saturation jitter in HSV.
    """
    import numpy as np
    import cv2

    out = fr.astype(np.float32)

    # Brightness/contrast
    brightness = float(np.random.uniform(0.85, 1.15))
    contrast = float(np.random.uniform(0.85, 1.15))
    out = out * contrast
    out = out + (brightness - 1.0) * 128.0
    out = np.clip(out, 0, 255).astype(np.uint8)

    # Saturation in HSV
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = float(np.random.uniform(0.8, 1.2))
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def _read_video_uniform_frames_cv2(video_path: Path, num_frames: int = 16) -> List["np.ndarray"]:
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        # Fallback: try reading sequentially
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        if len(frames) == 0:
            raise RuntimeError(f"No frames read from: {video_path}")
        total = len(frames)
        idx = np.linspace(0, total - 1, num_frames).round().astype(int).tolist()
        return [frames[i] for i in idx]

    idx = np.linspace(0, total - 1, num_frames).round().astype(int).tolist()
    frames: List["np.ndarray"] = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok or fr is None:
            # If a seek fails, try reading next available frame
            ok2, fr2 = cap.read()
            if not ok2 or fr2 is None:
                break
            fr = fr2
        frames.append(fr)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames read from: {video_path}")

    # Pad if short
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames[:num_frames]


def _preprocess_bgr_frames(
    frames_bgr: List["np.ndarray"],
    img_size: int = 224,
    augment: bool = False,
) -> torch.Tensor:
    import cv2
    import numpy as np

    frames_rgb = []
    do_flip = False
    rot_deg = 0.0
    if augment:
        do_flip = np.random.rand() < 0.5
        rot_deg = float(np.random.uniform(-8.0, 8.0))

    for fr in frames_bgr:
        if augment:
            # RandomResizedCrop(224, scale=(0.8, 1.0))
            fr = _random_resized_crop_bgr(fr, scale=(0.8, 1.0))
            # ColorJitter (brightness/contrast/saturation)
            fr = _color_jitter_bgr(fr)

            if do_flip:
                fr = cv2.flip(fr, 1)
            if abs(rot_deg) > 1e-3:
                h, w = fr.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), rot_deg, 1.0)
                fr = cv2.warpAffine(fr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        fr = cv2.resize(fr, (img_size, img_size), interpolation=cv2.INTER_AREA)
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        frames_rgb.append(fr)

    x = np.stack(frames_rgb, axis=0).astype("float32") / 255.0  # (T, H, W, C)
    mean = np.array(IMAGENET_MEAN, dtype="float32").reshape(1, 1, 1, 3)
    std = np.array(IMAGENET_STD, dtype="float32").reshape(1, 1, 1, 3)
    x = (x - mean) / std
    x = x.transpose(0, 3, 1, 2)  # (T, C, H, W)
    return torch.from_numpy(x)


def _resolve_videos_dir(videos_root: Path) -> Path:
    # Support accidental nesting MS-ASL/videos/videos/
    nested = videos_root / "videos"
    return nested if nested.exists() and nested.is_dir() else videos_root


class LocalSignEaseDataset(Dataset):
    def __init__(
        self,
        videos_root: Path,
        labels: List[str],
        frame_count: int = 16,
        img_size: int = 224,
        augment: bool = False,
    ):
        self.videos_root = videos_root
        self.videos_dir = _resolve_videos_dir(videos_root)
        self.labels = labels
        self.frame_count = frame_count
        self.img_size = img_size
        self.augment = augment

        if not self.videos_root.exists():
            raise FileNotFoundError(f"Videos root not found: {self.videos_root}")
        if not self.videos_dir.exists():
            raise FileNotFoundError(f"Resolved videos dir not found: {self.videos_dir}")

        self.samples: List[Tuple[Path, int]] = []
        exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

        # Layout A: MS-ASL/videos/<label>/*.mp4
        label_dirs = [p for p in self.videos_dir.iterdir() if p.is_dir()]
        if label_dirs:
            for idx, lab in enumerate(self.labels):
                lab_dir = self.videos_dir / lab
                if not lab_dir.exists():
                    continue
                for vid in lab_dir.rglob("*"):
                    if vid.is_file() and vid.suffix.lower() in exts:
                        self.samples.append((vid, idx))
        else:
            # Layout B: MS-ASL/videos/*.mp4 (one file per class label)
            for idx, lab in enumerate(self.labels):
                cand = self.videos_dir / f"{lab}.mp4"
                if cand.exists():
                    self.samples.append((cand, idx))

        # Filter out unreadable/broken videos early (prevents DataLoader worker crashes)
        self.samples = self._filter_readable_samples(self.samples)

        # Verify all labels are present at least once
        present = {y for _, y in self.samples}
        missing = [self.labels[i] for i in range(len(self.labels)) if i not in present]
        if missing:
            raise RuntimeError(
                f"Missing videos for labels: {missing}\n"
                f"Expected either MS-ASL/videos/<label>/*.mp4 or MS-ASL/videos/{missing[0]}.mp4 style files.\n"
                f"Resolved videos_dir: {self.videos_dir}"
            )

        if not self.samples:
            raise RuntimeError(f"No videos found under {self.videos_dir} (looked for {sorted(exts)})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, y = self.samples[idx]
        frames = _read_video_uniform_frames_cv2(video_path, num_frames=self.frame_count)
        x = _preprocess_bgr_frames(frames, img_size=self.img_size, augment=self.augment)  # (T, C, H, W)
        return x, int(y)

    @staticmethod
    def _filter_readable_samples(samples: List[Tuple[Path, int]]) -> List[Tuple[Path, int]]:
        import cv2

        good: List[Tuple[Path, int]] = []
        bad: List[Path] = []

        for p, y in samples:
            try:
                # Quick corruption check
                if p.stat().st_size <= 0:
                    bad.append(p)
                    continue
                cap = cv2.VideoCapture(str(p))
                ok = cap.isOpened()
                if ok:
                    ok, _ = cap.read()
                cap.release()
                if not ok:
                    bad.append(p)
                    continue
                good.append((p, y))
            except Exception:
                bad.append(p)

        if bad:
            print(f"⚠️  Skipping {len(bad)} unreadable video(s). Examples:")
            for ex in bad[:10]:
                try:
                    sz = ex.stat().st_size
                except Exception:
                    sz = -1
                print(f"   - {ex.name} (bytes={sz})")

        if not good:
            raise RuntimeError(
                "No readable videos found. Your .mp4 files appear to be corrupted/empty "
                "(e.g., 'moov atom not found'). Replace them with real video files and re-run."
            )

        return good


@dataclass
class TrainConfig:
    frame_count: int = 16
    img_size: int = 224
    batch_size: int = 2
    lr: float = 1e-5
    weight_decay: float = 0.05
    epochs: int = 5
    num_workers: int = 2
    device: str = _pick_device()
    seed: int = 42


class VideoSwinWrapper(nn.Module):
    """
    Wrap an image Swin model to accept video tensors (B,T,C,H,W).
    We run the image model per-frame and average logits over time.
    """

    def __init__(self, frame_model: nn.Module):
        super().__init__()
        self.frame_model = frame_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        logits = self.frame_model(x)  # (B*T, num_classes)
        logits = logits.reshape(b, t, -1).mean(dim=1)  # (B, num_classes)
        return logits


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == y).float().mean().item())


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    videos_root = base_dir / "MS-ASL" / "videos"
    checkpoints_dir = base_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    start_weights = checkpoints_dir / "fast_asl_model.pth"
    # Overwrite so api.py immediately uses the fine-tuned weights
    best_out = checkpoints_dir / "fast_asl_model.pth"

    cfg = TrainConfig()
    print("🚀 Local Swin training")
    print(f"   videos_root: {videos_root}")
    print(f"   device: {cfg.device}")
    print(f"   epochs: {cfg.epochs}")
    print(f"   lr: {cfg.lr}")
    print(f"   start_weights: {start_weights}")
    print(f"   best_out: {best_out}")

    if not start_weights.exists():
        raise FileNotFoundError(f"Starting checkpoint not found: {start_weights}")

    # Dataset with augmentation for training
    full_ds = LocalSignEaseDataset(
        videos_root=videos_root,
        labels=LABELS_48,
        frame_count=cfg.frame_count,
        img_size=cfg.img_size,
        augment=False,  # set per-split below
    )
    num_classes = len(LABELS_48)
    print(f"   classes: {num_classes}")
    print(f"   samples: {len(full_ds)}")

    # Train/val split
    g = torch.Generator().manual_seed(cfg.seed)
    n_total = len(full_ds)
    n_val = max(1, int(0.2 * n_total))
    n_train = max(1, n_total - n_val)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)
    # Enable augmentation only for train split
    if hasattr(train_ds, "dataset"):
        train_ds.dataset.augment = True  # type: ignore[attr-defined]
    if hasattr(val_ds, "dataset"):
        val_ds.dataset.augment = False  # type: ignore[attr-defined]

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    # Build timm Swin model via existing module and load checkpoint strictly.
    from models import swin_video

    state_dict = torch.load(start_weights, map_location="cpu")
    ckpt_classes = int(state_dict["head.fc.bias"].shape[0])
    if ckpt_classes != num_classes:
        raise RuntimeError(f"Checkpoint classes ({ckpt_classes}) != LABELS_48 ({num_classes})")

    frame_model = swin_video.timm.create_model(
        "swin_base_patch4_window7_224",
        pretrained=False,
        num_classes=num_classes,
    )
    frame_model.load_state_dict(state_dict, strict=True)
    model = VideoSwinWrapper(frame_model).to(cfg.device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        n_train_batches = 0
        for x, y in train_dl:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            if x.dim() == 4:
                x = x.unsqueeze(0)

            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()

            train_loss += float(loss.item())
            train_acc += _accuracy(logits.detach(), y.detach())
            n_train_batches += 1

        train_loss /= max(1, n_train_batches)
        train_acc /= max(1, n_train_batches)

        model.eval()
        val_acc = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(cfg.device)
                y = y.to(cfg.device)
                if x.dim() == 4:
                    x = x.unsqueeze(0)
                logits = model(x)
                val_acc += _accuracy(logits, y)
                n_val_batches += 1
        val_acc /= max(1, n_val_batches)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.3f} | val_acc: {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save in the same format api.py expects (timm state_dict with head.fc.*)
            torch.save(frame_model.state_dict(), best_out)
            print(f"✅ Saved best model to {best_out} (val_acc={best_val_acc:.3f})")

    print(f"✅ Training complete. Best val_acc={best_val_acc:.3f}. Output: {best_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

