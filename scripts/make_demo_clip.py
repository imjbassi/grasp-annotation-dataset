#!/usr/bin/env python3
"""Generate a tiny synthetic robot-arm 'grasp' clip for smoke-testing.

This lets you exercise the full frames -> upload -> export pipeline without
downloading the (large) BridgeData V2 archive. It renders a simple animation of
a two-finger gripper descending onto a block, written as an image sequence
under data/raw/demo_clip/.

Usage
-----
    python scripts/make_demo_clip.py --frames 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grasp_annot.config import RAW_DIR, ensure_dirs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--size", type=int, default=480)
    args = parser.parse_args()

    import cv2
    import numpy as np

    ensure_dirs()
    out_dir = RAW_DIR / "demo_clip"
    out_dir.mkdir(parents=True, exist_ok=True)

    w = h = args.size
    block_x, block_y = w // 2, int(h * 0.7)

    for i in range(args.frames):
        img = np.full((h, w, 3), 30, dtype=np.uint8)  # dark table
        # Object (block).
        cv2.rectangle(img, (block_x - 30, block_y - 30),
                      (block_x + 30, block_y + 30), (40, 120, 220), -1)
        # Gripper descends over time.
        t = i / max(args.frames - 1, 1)
        gy = int(h * 0.1 + t * (block_y - 40 - h * 0.1))
        gap = int(45 - t * 15)  # fingers close as it descends
        cv2.rectangle(img, (block_x - 6, 0), (block_x + 6, gy), (200, 200, 200), -1)
        cv2.rectangle(img, (block_x - gap - 6, gy), (block_x - gap + 2, gy + 40),
                      (230, 230, 230), -1)
        cv2.rectangle(img, (block_x + gap - 2, gy), (block_x + gap + 6, gy + 40),
                      (230, 230, 230), -1)
        cv2.imwrite(str(out_dir / f"frame_{i:04d}.png"), img)

    print(f"[demo] wrote {args.frames} frames -> {out_dir}")
    print("[demo] next: python scripts/extract_frames.py --every 3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
