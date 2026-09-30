#!/usr/bin/env python3
"""Render a short grasp-simulation clip (MP4 + GIF) for the README.

Simulates a planar two-link manipulator with a parallel-jaw gripper executing
a top-down grasp — approach, pre-shape, contact, lift — and renders it as a
white-background, publication-style figure: workspace panel with metric axes,
live gripper-aperture and end-effector-height traces, and the project's three
annotation types (``grasp_event`` box, ``object_contact`` keypoint,
``failure_mode`` label) overlaid the moment they become defined.

The joint angles come from closed-form inverse kinematics and the overlay
geometry is the same geometry written to the matching COCO record, so the clip
is a faithful visualisation of the schema — no robot, GPU, or simulator.

Usage
-----
    python scripts/render_grasp_sim.py            # -> assets/grasp_sim.{mp4,gif}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import imageio.v2 as imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grasp_annot.config import EXPORTS_DIR, ROOT_DIR, ensure_dirs  # noqa: E402

ASSETS_DIR = ROOT_DIR / "assets"

# --- render settings ---------------------------------------------------------
FPS = 30
DURATION = 9.0                     # seconds
W_IN, H_IN, DPI = 12.8, 7.2, 100   # 1280 x 720
GIF_EVERY, GIF_WIDTH = 2, 640      # 15 fps, half resolution

# --- palette (white-background, print-friendly) ------------------------------
INK = "#1a1a1a"
FAINT = "#8a8a8a"
LINK = "#37474f"
GRIP = "#263238"
JOINT_EDGE = "#263238"
BLOCK_FACE = "#b7cfe8"
BLOCK_EDGE = "#2b5b8a"
TRACE = "#b8b8b8"
BBOX = "#c0392b"        # grasp_event
KEYPT = "#1e7d32"       # object_contact
CURVE_A = "#2b5b8a"
CURVE_Z = "#8a5a2b"

# --- world geometry (metres) --------------------------------------------------
BASE = (0.10, 0.35)                # shoulder joint (on a pedestal)
L1, L2 = 0.32, 0.30                # link lengths
F_LEN = 0.075                      # finger length
PALM_OFF = 0.022                   # palm bar to wrist joint
F_HALF = 0.006                     # finger half-thickness (geometric)
GAP_OPEN = 0.078                   # half-aperture, fully open

BLOCK_CX, BLOCK_HW, BLOCK_H = 0.62, 0.035, 0.070
GAP_CLOSED = BLOCK_HW + F_HALF     # fingers touching the block faces

P_HOME = (0.50, 0.260)             # gripper midpoint at fingertip level
P_PRE = (BLOCK_CX, 0.260)
P_GRASP = (BLOCK_CX, 0.012)
P_LIFT = (BLOCK_CX, 0.240)

# (name, roman, t_start, t_end, p_from, p_to)
PHASES = [
    ("approach", "I", 0.0, 2.2, P_HOME, P_PRE),
    ("pre-shape", "II", 2.2, 3.8, P_PRE, P_GRASP),
    ("grasp", "III", 3.8, 5.2, P_GRASP, P_GRASP),
    ("lift", "IV", 5.2, 7.2, P_GRASP, P_LIFT),
    ("hold", "V", 7.2, DURATION, P_LIFT, P_LIFT),
]
T_CLOSE0, T_CLOSE1 = 3.8, 4.8      # finger closing window
T_BBOX = 5.0                       # grasp_event box appears
T_LABEL = 7.2                      # failure_mode label appears


def minjerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))


def gripper_state(t: float):
    """Return (phase_name, roman, p, half_gap) at time t."""
    name, roman, p = PHASES[-1][0], PHASES[-1][1], np.array(PHASES[-1][5])
    for pname, prom, t0, t1, a, b in PHASES:
        if t <= t1:
            s = minjerk((t - t0) / (t1 - t0))
            p = (1 - s) * np.array(a) + s * np.array(b)
            name, roman = pname, prom
            break
    s = minjerk((t - T_CLOSE0) / (T_CLOSE1 - T_CLOSE0))
    gap = (1 - s) * GAP_OPEN + s * GAP_CLOSED
    return name, roman, p, gap


def ik_elbow_up(target):
    """Closed-form 2-link IK; picks the solution whose elbow sits higher."""
    dx, dy = target[0] - BASE[0], target[1] - BASE[1]
    r = np.clip(np.hypot(dx, dy), abs(L1 - L2) + 1e-9, L1 + L2 - 1e-9)
    c2 = np.clip((r * r - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1.0, 1.0)
    best = None
    for th2 in (np.arccos(c2), -np.arccos(c2)):
        th1 = np.arctan2(dy, dx) - np.arctan2(L2 * np.sin(th2),
                                              L1 + L2 * np.cos(th2))
        elbow = (BASE[0] + L1 * np.cos(th1), BASE[1] + L1 * np.sin(th1))
        if best is None or elbow[1] > best[0][1]:
            best = (elbow, th1, th2)
    return best[0]  # elbow position


# --- figure ------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
})


def make_figure():
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor="white")
    ax = fig.add_axes([0.050, 0.115, 0.575, 0.760])           # workspace
    ax_a = fig.add_axes([0.700, 0.575, 0.265, 0.235])         # aperture
    ax_z = fig.add_axes([0.700, 0.255, 0.265, 0.235])         # height
    return fig, ax, ax_a, ax_z


def draw_workspace(ax, t, phase, roman, p, gap, contact, trace, block_c):
    ax.cla()
    ax.set_facecolor("white")
    ax.set_xlim(0.0, 0.96)
    ax.set_ylim(-0.075, 0.64)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.5, color="#d9d9d9", zorder=0)
    ax.set_xlabel("x [m]", fontsize=9.5)
    ax.set_ylabel("z [m]", fontsize=9.5)
    ax.tick_params(labelsize=8.5)

    # table + ground hatching
    ax.plot([0.02, 0.90], [0, 0], color=INK, lw=1.3, zorder=2)
    for hx in np.arange(0.03, 0.90, 0.028):
        ax.plot([hx, hx - 0.016], [0, -0.022], color=FAINT, lw=0.6, zorder=2)

    # pedestal + base
    ax.add_patch(plt.Rectangle((BASE[0] - 0.032, 0), 0.064, BASE[1],
                               facecolor="#e8e8e8", edgecolor=INK, lw=0.9,
                               zorder=3))

    # object block
    bx, by = block_c
    ax.add_patch(plt.Rectangle((bx - BLOCK_HW, by), 2 * BLOCK_HW, BLOCK_H,
                               facecolor=BLOCK_FACE, edgecolor=BLOCK_EDGE,
                               lw=1.2, zorder=4))

    # end-effector trace
    if len(trace) > 1:
        tr = np.asarray(trace)
        ax.plot(tr[:, 0], tr[:, 1], linestyle=(0, (1, 2)), lw=1.0,
                color=TRACE, zorder=3)

    # arm
    wrist = (p[0], p[1] + F_LEN + PALM_OFF)
    elbow = ik_elbow_up(wrist)
    ax.plot([BASE[0], elbow[0]], [BASE[1], elbow[1]], color=LINK, lw=7,
            solid_capstyle="round", zorder=5)
    ax.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], color=LINK, lw=6,
            solid_capstyle="round", zorder=5)
    for j, r in ((BASE, 0.016), (elbow, 0.013), (wrist, 0.011)):
        ax.add_patch(plt.Circle(j, r, facecolor="white", edgecolor=JOINT_EDGE,
                                lw=1.4, zorder=6))

    # gripper: wrist stem, palm bar, two fingers
    palm_y = p[1] + F_LEN
    ax.plot([p[0], p[0]], [wrist[1], palm_y], color=GRIP, lw=5,
            solid_capstyle="round", zorder=5)
    ax.plot([p[0] - gap, p[0] + gap], [palm_y, palm_y], color=GRIP, lw=5.5,
            solid_capstyle="round", zorder=5)
    for fx in (p[0] - gap, p[0] + gap):
        ax.plot([fx, fx], [palm_y, p[1]], color=GRIP, lw=4.5,
                solid_capstyle="round", zorder=5)

    # annotation overlays -----------------------------------------------------
    kp = (bx - BLOCK_HW, by + BLOCK_H - 0.018)
    if contact:
        ax.plot(*kp, marker="o", ms=7, mfc=KEYPT, mec="white", mew=1.2,
                zorder=8)
        ax.annotate("object_contact", kp, xytext=(kp[0] - 0.028, kp[1] + 0.012),
                    ha="right", va="bottom", fontsize=8.5, color=KEYPT,
                    family="monospace", zorder=8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75,
                              pad=1))
    box = None
    if t >= T_BBOX:
        pad = 0.014
        x0 = p[0] - gap - F_HALF - pad
        y0 = by - pad
        bw = 2 * (gap + F_HALF + pad)
        bh = (palm_y + pad) - y0
        box = (x0, y0, bw, bh)
        ax.add_patch(plt.Rectangle((x0, y0), bw, bh, fill=False,
                                   edgecolor=BBOX, lw=1.4,
                                   linestyle=(0, (5, 3)), zorder=7))
        ax.text(x0 + bw + 0.010, y0 + bh, "grasp_event", fontsize=8.5,
                color=BBOX, family="monospace", va="top", zorder=8)
    if t >= T_LABEL:
        ax.text(bx, by - 0.048, "failure_mode = success", fontsize=9,
                color=KEYPT, family="monospace", ha="center", zorder=8)

    # phase + clock
    ax.text(0.02, 0.975, f"Phase {roman} — {phase}", transform=ax.transAxes,
            fontsize=10, color=INK, va="top")
    ax.text(0.98, 0.975, f"t = {t:4.2f} s", transform=ax.transAxes,
            fontsize=10, color=INK, va="top", ha="right", family="monospace")
    return kp, box, wrist, elbow


def draw_traces(ax_a, ax_z, times, gaps, heights):
    for ax, ys, col, ylab, ylim in (
        (ax_a, gaps, CURVE_A, "aperture [mm]", (55, 175)),
        (ax_z, heights, CURVE_Z, "eef height z [m]", (-0.02, 0.34)),
    ):
        ax.cla()
        ax.set_facecolor("white")
        ax.set_xlim(0, DURATION)
        ax.set_ylim(*ylim)
        ax.grid(True, linestyle=":", linewidth=0.5, color="#d9d9d9")
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.tick_params(labelsize=8)
        for _, _, t0, _, _, _ in PHASES[1:]:
            ax.axvline(t0, color="#dddddd", lw=0.7)
        if len(times) > 1:
            ax.plot(times, ys, color=col, lw=1.4)
        ax.plot(times[-1], ys[-1], marker="o", ms=4, color=col)
    ax_a.set_xticklabels([])
    ax_z.set_xlabel("t [s]", fontsize=8.5)


def draw_chrome(fig, t, phase, roman, gap, contact):
    for txt in list(fig.texts):
        txt.remove()
    fig.text(0.050, 0.952, "Parallel-Jaw Grasp Acquisition — Planar Simulation",
             fontsize=16, color=INK)
    fig.text(0.050, 0.918, "Two-link manipulator, closed-form inverse kinematics"
             "  ·  annotation schema overlay: grasp_event / object_contact / "
             "failure_mode", fontsize=9, color=FAINT)
    # state read-out
    rows = [
        ("t", f"{t:5.2f} s"),
        ("phase", f"{roman} · {phase}"),
        ("aperture", f"{2 * gap * 1e3:5.1f} mm"),
        ("contact", "yes" if contact else "no"),
        ("failure_mode", "success" if t >= T_LABEL else "—"),
    ]
    fig.text(0.700, 0.185, "Simulation state", fontsize=9.5, color=INK)
    for i, (k, v) in enumerate(rows):
        y = 0.155 - i * 0.026
        fig.text(0.700, y, k, fontsize=8.5, color=FAINT, family="monospace")
        fig.text(0.810, y, v, fontsize=8.5, color=INK, family="monospace")
    fig.text(0.050, 0.030,
             "Fig. 1.  Simulated top-down parallel-jaw grasp: approach, "
             "pre-shape, contact, and lift. Overlays show the three annotation "
             "types this repository exports to COCO.",
             fontsize=9, color=INK)


def render_frame(fig, ax, ax_a, ax_z, t, times, gaps, heights, trace,
                 block_c):
    phase, roman, p, gap = gripper_state(t)
    contact = gap <= GAP_CLOSED + 1e-4
    kp, box, _, _ = draw_workspace(ax, t, phase, roman, p, gap, contact,
                                   trace, block_c)
    draw_traces(ax_a, ax_z, times, gaps, heights)
    draw_chrome(fig, t, phase, roman, gap, contact)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    return buf, p, gap, contact, kp, box


def to_pixels(ax, xy):
    """Data coordinates -> image pixel coordinates (origin top-left)."""
    px, py = ax.transData.transform(xy)
    return float(px), float(H_IN * DPI - py)


def write_coco(ax, kp, box):
    """Write a COCO record matching the final overlay geometry, in pixels."""
    x0, y0 = to_pixels(ax, (box[0], box[1] + box[3]))
    x1, y1 = to_pixels(ax, (box[0] + box[2], box[1]))
    kx, ky = to_pixels(ax, kp)
    bbox = [round(v, 1) for v in (x0, y0, x1 - x0, y1 - y0)]
    coco = {
        "info": {"description": "Robot grasp annotations (simulation demo)",
                 "version": "1.0"},
        "categories": [
            {"id": 1, "name": "grasp_event", "supercategory": "grasp"},
            {"id": 2, "name": "object_contact", "supercategory": "grasp",
             "keypoints": ["contact_point"], "skeleton": []},
        ],
        "images": [{
            "id": 1, "file_name": "grasp_sim__f{:06d}.png".format(
                int(DURATION * FPS) - 1),
            "width": int(W_IN * DPI), "height": int(H_IN * DPI),
            "failure_mode": "success",
        }],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": bbox,
             "area": round(bbox[2] * bbox[3], 1), "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2,
             "keypoints": [round(kx, 1), round(ky, 1), 2], "num_keypoints": 1,
             "bbox": [round(kx, 1), round(ky, 1), 0, 0], "area": 0,
             "iscrowd": 0},
        ],
    }
    out = EXPORTS_DIR / "sim_grasp_coco.json"
    out.write_text(json.dumps(coco, indent=2))
    return out


def main() -> int:
    ensure_dirs()
    ASSETS_DIR.mkdir(exist_ok=True)
    mp4_path = ASSETS_DIR / "grasp_sim.mp4"
    gif_path = ASSETS_DIR / "grasp_sim.gif"

    fig, ax, ax_a, ax_z = make_figure()
    n_frames = int(DURATION * FPS)
    times, gaps, heights, trace = [], [], [], []
    gif_frames = []
    block_c = [BLOCK_CX, 0.0]
    attach = None
    last_kp = last_box = None

    writer = imageio.get_writer(str(mp4_path), fps=FPS, codec="libx264",
                                quality=8, pixelformat="yuv420p")
    for i in range(n_frames):
        t = i / FPS
        _, _, p, gap = gripper_state(t)
        contact = gap <= GAP_CLOSED + 1e-4
        if contact and attach is None:
            attach = block_c[1] - p[1]           # grab the block where it sits
        if attach is not None:
            block_c = [p[0], p[1] + attach]
        times.append(t)
        gaps.append(2 * gap * 1e3)
        heights.append(p[1])
        trace.append(tuple(p))

        buf, _, _, contact, kp, box = render_frame(
            fig, ax, ax_a, ax_z, t, times, gaps, heights, trace, block_c)
        writer.append_data(buf)
        if box is not None:
            last_kp, last_box = kp, box
        if i % GIF_EVERY == 0:
            im = Image.fromarray(buf)
            im = im.resize((GIF_WIDTH, GIF_WIDTH * buf.shape[0]
                            // buf.shape[1]), Image.LANCZOS)
            gif_frames.append(im.quantize(colors=128,
                                          method=Image.Quantize.MEDIANCUT))
    writer.close()

    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                       duration=int(1000 * GIF_EVERY / FPS), loop=0,
                       optimize=True)
    coco_path = write_coco(ax, last_kp, last_box)
    plt.close(fig)

    print(f"[sim] {mp4_path}")
    print(f"[sim] {gif_path}")
    print(f"[sim] {coco_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
