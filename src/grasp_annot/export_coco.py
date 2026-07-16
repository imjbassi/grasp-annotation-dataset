"""Export completed Labelbox labels and serialise them to COCO-format JSON.

Pulls every label in the project via the Labelbox export API, then converts:

    grasp_event     (bbox)     -> COCO object annotation with ``bbox``
    object_contact  (keypoint) -> COCO object annotation with ``keypoints``
    failure_mode    (radio)    -> stored on the *image* record as
                                  ``failure_mode`` (image-level attribute)

The result is written to ``data/exports/grasp_coco.json`` and validates against
the standard COCO object-detection / keypoint schema, so it can be loaded with
``pycocotools`` or any COCO-aware training pipeline.

Usage
-----
    python -m grasp_annot.export_coco
    python -m grasp_annot.export_coco --out data/exports/grasp_coco.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .config import EXPORTS_DIR, Settings, ensure_dirs
from .ontology import FAILURE_MODE, GRASP_EVENT, OBJECT_CONTACT

# COCO category ids are 1-indexed by convention.
CATEGORIES = [
    {"id": 1, "name": GRASP_EVENT, "supercategory": "grasp"},
    {
        "id": 2,
        "name": OBJECT_CONTACT,
        "supercategory": "grasp",
        "keypoints": ["contact_point"],
        "skeleton": [],
    },
]
_CAT_ID = {c["name"]: c["id"] for c in CATEGORIES}


def _stream_export(project) -> Iterable[Dict[str, Any]]:
    """Yield one export record per data row, across SDK versions.

    Newer SDKs expose ``project.export()`` (streaming); older ones expose
    ``project.export_v2()`` (buffered result list). We support both.
    """
    params = {
        "attachments": False,
        "metadata_fields": False,
        "data_row_details": True,
        "project_details": True,
        "label_details": True,
    }
    if hasattr(project, "export"):
        task = project.export(params=params)
        task.wait_till_done()
        buffer: List[Dict[str, Any]] = []
        # stream_handlers differ slightly by version; fall back to result.
        try:
            task.get_buffered_stream()  # type: ignore[attr-defined]
            for out in task.get_buffered_stream():  # type: ignore[attr-defined]
                buffer.append(out.json)
            if buffer:
                yield from buffer
                return
        except Exception:
            pass
        for rec in task.result or []:
            yield rec
        return

    task = project.export_v2(params=params)  # type: ignore[attr-defined]
    task.wait_till_done()
    if task.errors:
        print(f"[warn] export reported errors: {task.errors}", file=sys.stderr)
    yield from (task.result or [])


def _iter_labels(record: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield each label object from an export record, tolerating shape drift."""
    projects = record.get("projects", {})
    for proj in projects.values():
        for label in proj.get("labels", []):
            yield label


def _bbox_to_coco(box: Dict[str, Any]) -> List[float]:
    b = box["bounding_box"]
    return [float(b["left"]), float(b["top"]), float(b["width"]), float(b["height"])]


def convert(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert Labelbox export records into a COCO dict."""
    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    ann_id = 1

    for img_id, record in enumerate(records, start=1):
        data_row = record.get("data_row", {})
        media = record.get("media_attributes", {}) or {}
        global_key = data_row.get("global_key") or data_row.get("id")

        image_rec = {
            "id": img_id,
            "file_name": global_key,
            "labelbox_data_row_id": data_row.get("id"),
            "width": media.get("width"),
            "height": media.get("height"),
            "failure_mode": None,  # filled in from the radio classification
        }

        for label in _iter_labels(record):
            ann_block = label.get("annotations", {})

            # Objects: bbox (grasp_event) and keypoint (object_contact).
            for obj in ann_block.get("objects", []):
                name = obj.get("name")
                if name == GRASP_EVENT and "bounding_box" in obj:
                    bbox = _bbox_to_coco(obj)
                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": _CAT_ID[GRASP_EVENT],
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1
                elif name == OBJECT_CONTACT and "point" in obj:
                    px, py = float(obj["point"]["x"]), float(obj["point"]["y"])
                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": _CAT_ID[OBJECT_CONTACT],
                            "keypoints": [px, py, 2],  # 2 == labeled & visible
                            "num_keypoints": 1,
                            "bbox": [px, py, 0, 0],
                            "area": 0,
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1

            # Image-level classification: failure_mode radio answer.
            for cls in ann_block.get("classifications", []):
                if cls.get("name") == FAILURE_MODE:
                    answer = cls.get("radio_answer") or {}
                    image_rec["failure_mode"] = answer.get("value") or answer.get("name")

        images.append(image_rec)

    return {
        "info": {
            "description": "Robot manipulation grasp annotations (Labelbox export)",
            "version": "1.0",
            "date_created": datetime.now(timezone.utc).isoformat(),
        },
        "licenses": [],
        "categories": CATEGORIES,
        "images": images,
        "annotations": annotations,
    }


def export(settings: Settings, out_path: Path) -> Dict[str, Any]:
    import labelbox as lb  # lazy: only needed when actually talking to Labelbox

    from .create_project import _find_existing_project

    client = lb.Client(api_key=settings.api_key)
    project = _find_existing_project(client, settings.project_name)
    if project is None:
        raise RuntimeError(
            f"Project '{settings.project_name}' not found. Nothing to export."
        )

    print(f"[export] pulling labels from '{project.name}' ...")
    coco = convert(_stream_export(project))

    ensure_dirs()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco, indent=2))
    print(
        f"[done]   {len(coco['images'])} images, "
        f"{len(coco['annotations'])} annotations -> {out_path}"
    )
    return coco


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=EXPORTS_DIR / "grasp_coco.json",
        help="Output path for the COCO JSON file.",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    export(settings, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
