"""Offline unit tests for the COCO converter.

These run with no Labelbox account and no network — they feed a synthetic
export record (matching Labelbox's export JSON shape) through ``convert`` and
assert the resulting COCO structure is well formed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grasp_annot.export_coco import convert  # noqa: E402


def _fake_record():
    return {
        "data_row": {"id": "dr_1", "global_key": "clip_a__f000010.jpg"},
        "media_attributes": {"width": 640, "height": 480},
        "projects": {
            "proj_1": {
                "labels": [
                    {
                        "annotations": {
                            "objects": [
                                {
                                    "name": "grasp_event",
                                    "bounding_box": {
                                        "top": 100, "left": 50,
                                        "height": 40, "width": 30,
                                    },
                                },
                                {
                                    "name": "object_contact",
                                    "point": {"x": 65, "y": 120},
                                },
                            ],
                            "classifications": [
                                {
                                    "name": "failure_mode",
                                    "radio_answer": {"value": "drop", "name": "drop"},
                                }
                            ],
                        }
                    }
                ]
            }
        },
    }


def test_convert_shapes():
    coco = convert([_fake_record()])

    assert {"info", "images", "annotations", "categories"} <= set(coco)
    assert len(coco["images"]) == 1
    img = coco["images"][0]
    assert img["file_name"] == "clip_a__f000010.jpg"
    assert img["width"] == 640 and img["height"] == 480
    assert img["failure_mode"] == "drop"


def test_convert_bbox_and_keypoint():
    coco = convert([_fake_record()])
    anns = coco["annotations"]
    assert len(anns) == 2

    bbox = next(a for a in anns if a["category_id"] == 1)
    assert bbox["bbox"] == [50.0, 100.0, 30.0, 40.0]
    assert bbox["area"] == 30.0 * 40.0

    kp = next(a for a in anns if a["category_id"] == 2)
    assert kp["keypoints"] == [65.0, 120.0, 2]
    assert kp["num_keypoints"] == 1


def test_convert_empty():
    coco = convert([])
    assert coco["images"] == []
    assert coco["annotations"] == []
