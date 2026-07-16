#!/usr/bin/env python3
"""Extract still frames from raw robot-arm clips into data/frames/.

Handles the two shapes the download script can produce:

  * Video files (.mp4/.avi/.mov) — sampled every N frames with OpenCV.
  * Image-sequence folders (BridgeData V2 stores trajectories as numbered
    .jpg/.png per timestep) — copied/downsampled directly.

Frames are named ``<clip>__f<index>.jpg`` so that the source clip and timestep
survive all the way through to the COCO export's ``file_name`` field.

Usage
-----
    python scripts/extract_frames.py --every 10
    python scripts/extract_frames.py --src data/raw/bridge --every 5 --max 500
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Make the src/ package importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grasp_annot.config import FRAMES_DIR, RAW_DIR, ensure_dirs  # noqa: E402

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _extract_video(path: Path, out_dir: Path, every: int, cap: int) -> int:
    try:
        import cv2
    except ImportError:
        print("opencv-python is required for video input: pip install opencv-python",
              file=sys.stderr)
        raise
    stem = path.stem
    cap_reader = cv2.VideoCapture(str(path))
    idx = written = 0
    while True:
        ok, frame = cap_reader.read()
        if not ok:
            break
        if idx % every == 0:
            out = out_dir / f"{stem}__f{idx:06d}.jpg"
            cv2.imwrite(str(out), frame)
            written += 1
            if cap and written >= cap:
                break
        idx += 1
    cap_reader.release()
    return written


def _extract_image_sequence(seq_dir: Path, out_dir: Path, every: int, cap: int) -> int:
    frames = sorted(p for p in seq_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    stem = seq_dir.name
    written = 0
    for i, frame in enumerate(frames):
        if i % every != 0:
            continue
        out = out_dir / f"{stem}__f{i:06d}{frame.suffix.lower()}"
        shutil.copy2(frame, out)
        written += 1
        if cap and written >= cap:
            break
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=RAW_DIR, help="Raw data dir.")
    parser.add_argument("--every", type=int, default=10, help="Keep every Nth frame.")
    parser.add_argument("--max", type=int, default=0, help="Cap frames per clip (0=all).")
    args = parser.parse_args()

    ensure_dirs()
    src: Path = args.src
    if not src.exists():
        print(f"Source {src} does not exist. Run scripts/download_dataset.sh first.",
              file=sys.stderr)
        return 1

    total = 0

    # 1) Any video files anywhere under src.
    for video in sorted(src.rglob("*")):
        if video.suffix.lower() in VIDEO_SUFFIXES:
            n = _extract_video(video, FRAMES_DIR, args.every, args.max)
            print(f"[frames] {video.name}: {n} frames")
            total += n

    # 2) Any folder that directly contains an image sequence.
    for folder in sorted(p for p in src.rglob("*") if p.is_dir()):
        imgs = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
        if len(imgs) >= 2:  # treat as a trajectory
            n = _extract_image_sequence(folder, FRAMES_DIR, args.every, args.max)
            if n:
                print(f"[frames] {folder.name}/: {n} frames")
                total += n

    print(f"\n[done] {total} frames written to {FRAMES_DIR}")
    if total == 0:
        print("[hint] No videos or image sequences found — check --src path.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
