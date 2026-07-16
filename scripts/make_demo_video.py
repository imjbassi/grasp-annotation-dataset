#!/usr/bin/env python3
"""Render an MP4 walkthrough of the grasp-annotation pipeline.

Produces data/exports/demo_walkthrough.mp4 — a narrated (captioned) demo that
shows a synthetic robot-arm grasp clip, the three annotation types overlaid
exactly as an annotator would place them (grasp_event bbox, object_contact
keypoint, failure_mode label), and the resulting COCO JSON. The overlay
coordinates are the same ones written to the COCO file, so the video is a
faithful visualisation of the real schema — no Labelbox account required.

Usage
-----
    python scripts/make_demo_video.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grasp_annot.config import EXPORTS_DIR, ensure_dirs  # noqa: E402

W, H = 1280, 720
FPS = 24
PANEL_X, PANEL_Y = 48, 150          # scene panel top-left on the canvas
SW, SH = 640, 480                   # scene (camera) dimensions
RED = (60, 60, 240)
GREEN = (90, 210, 90)
GREY = (200, 200, 200)
WHITE = (245, 245, 245)
DIM = (150, 150, 150)
BG = (28, 26, 24)


# --- scene geometry (mirrors scripts/make_demo_clip.py) ---------------------
BLOCK_CX, BLOCK_CY, BLOCK_HW = SW // 2, int(SH * 0.7), 34


def render_scene(t: float):
    """Return (scene_img, annotations) for progress t in [0, 1].

    annotations is a dict with bbox / keypoint in *scene* coordinates, or None
    while the gripper is still descending.
    """
    img = np.full((SH, SW, 3), 30, dtype=np.uint8)
    # table shading
    cv2.rectangle(img, (0, int(SH * 0.72)), (SW, SH), (48, 44, 40), -1)
    # object block
    cv2.rectangle(img, (BLOCK_CX - BLOCK_HW, BLOCK_CY - BLOCK_HW),
                  (BLOCK_CX + BLOCK_HW, BLOCK_CY + BLOCK_HW), (40, 120, 220), -1)
    cv2.rectangle(img, (BLOCK_CX - BLOCK_HW, BLOCK_CY - BLOCK_HW),
                  (BLOCK_CX + BLOCK_HW, BLOCK_CY + BLOCK_HW), (20, 80, 160), 2)

    gy = int(SH * 0.08 + t * (BLOCK_CY - BLOCK_HW - SH * 0.08))
    gap = int(48 - t * 16)
    # wrist / shaft
    cv2.rectangle(img, (BLOCK_CX - 7, 0), (BLOCK_CX + 7, gy), GREY, -1)
    # two fingers
    cv2.rectangle(img, (BLOCK_CX - gap - 7, gy), (BLOCK_CX - gap + 3, gy + 46), WHITE, -1)
    cv2.rectangle(img, (BLOCK_CX + gap - 3, gy), (BLOCK_CX + gap + 7, gy + 46), WHITE, -1)

    contact = gy + 46 >= BLOCK_CY - BLOCK_HW  # fingers reached the block top
    ann = None
    if contact:
        pad = 10
        left = BLOCK_CX - gap - 7 - pad
        top = gy - pad
        right = BLOCK_CX + gap + 7 + pad
        bottom = BLOCK_CY + BLOCK_HW + pad
        ann = {
            "bbox": [left, top, right - left, bottom - top],          # x,y,w,h
            "keypoint": [BLOCK_CX - BLOCK_HW, BLOCK_CY - BLOCK_HW + 4],  # contact pt
            "failure_mode": "success",
        }
    return img, ann


# --- drawing helpers --------------------------------------------------------
def put(canvas, text, org, scale=0.7, color=WHITE, thick=2):
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                cv2.LINE_AA)


def frame_canvas():
    c = np.full((H, W, 3), BG, dtype=np.uint8)
    put(c, "Robot Grasp Annotation  -  pipeline demo", (48, 60), 0.9, WHITE, 2)
    put(c, "github.com/imjbassi/grasp-annotation-dataset", (48, 92), 0.5, DIM, 1)
    return c


def draw_scene(canvas, scene, label):
    canvas[PANEL_Y:PANEL_Y + SH, PANEL_X:PANEL_X + SW] = scene
    cv2.rectangle(canvas, (PANEL_X - 2, PANEL_Y - 2),
                  (PANEL_X + SW + 2, PANEL_Y + SH + 2), (90, 90, 90), 2)
    put(canvas, label, (PANEL_X, PANEL_Y - 12), 0.6, DIM, 1)


def draw_overlay(canvas, ann, reveal):
    """reveal in {0:none, 1:bbox, 2:+keypoint, 3:+label}."""
    if ann is None:
        return
    ox, oy = PANEL_X, PANEL_Y
    if reveal >= 1:
        x, y, w, h = ann["bbox"]
        cv2.rectangle(canvas, (ox + x, oy + y), (ox + x + w, oy + y + h), RED, 2)
        put(canvas, "grasp_event", (ox + x, oy + y - 6), 0.5, RED, 1)
    if reveal >= 2:
        kx, ky = ann["keypoint"]
        cv2.circle(canvas, (ox + kx, oy + ky), 6, GREEN, -1)
        cv2.circle(canvas, (ox + kx, oy + ky), 9, GREEN, 1)
        put(canvas, "object_contact", (ox + kx + 12, oy + ky + 4), 0.5, GREEN, 1)
    if reveal >= 3:
        put(canvas, f"failure_mode: {ann['failure_mode']}", (ox, oy + SH + 28),
            0.6, GREEN, 2)


def draw_sidebar(canvas, lines, highlight=-1):
    x = PANEL_X + SW + 60
    put(canvas, "ANNOTATION SCHEMA", (x, PANEL_Y + 10), 0.6, WHITE, 2)
    for i, (name, desc, col) in enumerate(lines):
        y = PANEL_Y + 60 + i * 78
        active = i == highlight
        cv2.circle(canvas, (x + 8, y - 6), 7, col, -1 if active else 2)
        put(canvas, name, (x + 28, y), 0.62, WHITE if active else DIM, 2)
        put(canvas, desc, (x + 28, y + 26), 0.44, DIM, 1)


def draw_json(canvas, snippet_lines, shown):
    x = PANEL_X + SW + 60
    put(canvas, "data/exports/grasp_coco.json", (x, PANEL_Y + 10), 0.55, WHITE, 2)
    for i, line in enumerate(snippet_lines[:shown]):
        put(canvas, line, (x, PANEL_Y + 50 + i * 26), 0.44, (150, 220, 150), 1)


# --- timeline ---------------------------------------------------------------
def main() -> int:
    ensure_dirs()
    out_path = EXPORTS_DIR / "demo_walkthrough.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             FPS, (W, H))
    if not writer.isOpened():  # fallback codec
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"),
                                 FPS, (W, H))

    schema = [
        ("grasp_event", "bounding box  ->  COCO bbox", RED),
        ("object_contact", "keypoint  ->  COCO keypoints", GREEN),
        ("failure_mode", "radio  ->  image label", (90, 160, 240)),
    ]

    def hold(canvas, seconds):
        for _ in range(int(seconds * FPS)):
            writer.write(canvas)

    # Segment 0 — title card
    c = frame_canvas()
    put(c, "From raw robot video to a COCO-format grasp dataset", (48, 380),
        0.8, WHITE, 2)
    put(c, "download  ->  extract frames  ->  Labelbox ontology  ->  label  ->  export",
        (48, 430), 0.6, DIM, 1)
    hold(c, 2.2)

    # Segment 1 — raw clip descending
    steps = 48
    for i in range(steps):
        t = i / (steps - 1)
        scene, _ = render_scene(t * 0.72)  # stop before contact
        c = frame_canvas()
        draw_scene(c, scene, "Stage 1  -  raw robot-arm frame (BridgeData V2 / demo clip)")
        draw_sidebar(c, schema)
        writer.write(c)

    # Segment 2 — reach contact, reveal annotations one at a time
    scene, ann = render_scene(1.0)
    for reveal, hl, secs in [(0, -1, 0.4), (1, 0, 1.1), (2, 1, 1.1), (3, 2, 1.4)]:
        c = frame_canvas()
        draw_scene(c, scene, "Stage 2  -  human annotation")
        draw_overlay(c, ann, reveal)
        draw_sidebar(c, schema, highlight=hl)
        hold(c, secs)

    # Segment 3 — COCO export typing out
    coco_lines = [
        '"images": [{',
        f'  "file_name": "demo_clip__f000012.png",',
        f'  "width": {SW}, "height": {SH},',
        f'  "failure_mode": "{ann["failure_mode"]}"',
        '}],',
        '"annotations": [',
        f'  {{"category_id": 1, "bbox": {ann["bbox"]}}},',
        f'  {{"category_id": 2, "keypoints": {ann["keypoint"] + [2]}}}',
        '],',
        '"categories": [',
        '  {"id": 1, "name": "grasp_event"},',
        '  {"id": 2, "name": "object_contact"}]',
    ]
    for shown in range(len(coco_lines) + 1):
        c = frame_canvas()
        draw_scene(c, scene, "Stage 3  -  COCO export")
        draw_overlay(c, ann, 3)
        draw_json(c, coco_lines, shown)
        hold(c, 0.28)
    # linger on the full JSON
    c = frame_canvas()
    draw_scene(c, scene, "Stage 3  -  COCO export")
    draw_overlay(c, ann, 3)
    draw_json(c, coco_lines, len(coco_lines))
    hold(c, 2.0)

    # End card
    c = frame_canvas()
    put(c, "Runs with no robot, no GPU, no simulator.", (48, 360), 0.85, WHITE, 2)
    put(c, "make demo  ->  extract_frames  ->  create_project  ->  upload  ->  export",
        (48, 410), 0.58, DIM, 1)
    put(c, "github.com/imjbassi/grasp-annotation-dataset", (48, 470), 0.6, GREEN, 2)
    hold(c, 2.4)

    writer.release()

    # write a matching real COCO file so the video and JSON agree
    coco = {
        "info": {"description": "Robot grasp annotations (demo)", "version": "1.0"},
        "categories": [
            {"id": 1, "name": "grasp_event", "supercategory": "grasp"},
            {"id": 2, "name": "object_contact", "supercategory": "grasp",
             "keypoints": ["contact_point"], "skeleton": []},
        ],
        "images": [{
            "id": 1, "file_name": "demo_clip__f000012.png",
            "width": SW, "height": SH, "failure_mode": ann["failure_mode"],
        }],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": ann["bbox"],
             "area": ann["bbox"][2] * ann["bbox"][3], "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2,
             "keypoints": ann["keypoint"] + [2], "num_keypoints": 1,
             "bbox": ann["keypoint"] + [0, 0], "area": 0, "iscrowd": 0},
        ],
    }
    (EXPORTS_DIR / "demo_coco.json").write_text(json.dumps(coco, indent=2))

    print(f"[video] {out_path}")
    print(f"[coco]  {EXPORTS_DIR / 'demo_coco.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
