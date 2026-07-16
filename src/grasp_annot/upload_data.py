"""Upload extracted frames to Labelbox and batch them into the project.

Reads image frames from ``data/frames/`` (produced by
``scripts/extract_frames.py``), registers each as a data row in a Labelbox
dataset, then attaches them to the project as a labelling batch.

Local image files are uploaded directly by the SDK — no public URL or cloud
bucket is required, which keeps the whole pipeline free and hardware-free.

Usage
-----
    python -m grasp_annot.upload_data                 # upload everything
    python -m grasp_annot.upload_data --limit 50      # cap for a quick test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import labelbox as lb

from .config import FRAMES_DIR, Settings
from .create_project import _find_existing_project

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _collect_frames(frames_dir: Path, limit: int | None) -> List[Path]:
    frames = sorted(
        p for p in frames_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not frames:
        raise FileNotFoundError(
            f"No frames found under {frames_dir}. Run scripts/extract_frames.py "
            "first (see README)."
        )
    return frames[:limit] if limit else frames


def upload(settings: Settings, limit: int | None = None) -> dict:
    client = lb.Client(api_key=settings.api_key)

    project = _find_existing_project(client, settings.project_name)
    if project is None:
        raise RuntimeError(
            f"Project '{settings.project_name}' not found. "
            "Run `python -m grasp_annot.create_project` first."
        )

    dataset = client.create_dataset(name=settings.dataset_name)
    frames = _collect_frames(FRAMES_DIR, limit)
    print(f"[upload] {len(frames)} frames -> dataset '{dataset.name}'")

    # global_key = stable, human-readable id (relative path). Used later to
    # correlate the COCO export back to the source frame on disk.
    rows = []
    global_keys = []
    for frame in frames:
        gkey = str(frame.relative_to(FRAMES_DIR)).replace("/", "__")
        global_keys.append(gkey)
        rows.append(
            {
                "row_data": str(frame),
                "global_key": gkey,
                "media_type": "IMAGE",
            }
        )

    task = dataset.create_data_rows(rows)
    task.wait_till_done()
    if task.errors:
        print(f"[warn] {len(task.errors)} data rows failed to upload", file=sys.stderr)
        for err in task.errors[:5]:
            print(f"       {err}", file=sys.stderr)

    batch = project.create_batch(
        name=settings.batch_name,
        global_keys=global_keys,
        priority=5,
    )
    print(f"[batch]  '{batch.name}' -> {len(global_keys)} data rows queued for labelling")
    print(f"\nLabel them here:\n  https://app.labelbox.com/projects/{project.uid}")
    return {"dataset_id": dataset.uid, "batch_id": batch.uid, "count": len(global_keys)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Max frames to upload.")
    args = parser.parse_args()

    settings = Settings.from_env()
    upload(settings, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
