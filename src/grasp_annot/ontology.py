"""Declarative annotation schema for the grasp dataset.

This is the single source of truth for what annotators are asked to label. Both
the Labelbox project provisioning (``create_project.py``) and the COCO export
(``export_coco.py``) import these constants so the schema can never drift
between the two ends of the pipeline.

Schema
------
grasp_event    (bounding box)  A tight box around the gripper–object contact
                               region at the moment a grasp is attempted.
object_contact (keypoint)      A single point marking where a fingertip first
                               contacts the object surface.
failure_mode   (radio class)   Per-frame outcome label. Exactly one of:
                               drop | miss | collision | timeout | success
"""

from __future__ import annotations

from typing import List

# --- Tool / class names (also used as COCO category names) ------------------
GRASP_EVENT = "grasp_event"
OBJECT_CONTACT = "object_contact"
FAILURE_MODE = "failure_mode"

FAILURE_OPTIONS: List[str] = ["drop", "miss", "collision", "timeout", "success"]


def build_ontology():
    """Return a Labelbox ``OntologyBuilder`` describing the grasp schema."""
    import labelbox as lb  # lazy: keeps the schema constants importable offline

    return lb.OntologyBuilder(
        tools=[
            lb.Tool(
                tool=lb.Tool.Type.BBOX,
                name=GRASP_EVENT,
                color="#FF3B30",
            ),
            lb.Tool(
                tool=lb.Tool.Type.POINT,
                name=OBJECT_CONTACT,
                color="#34C759",
            ),
        ],
        classifications=[
            lb.Classification(
                class_type=lb.Classification.Type.RADIO,
                name=FAILURE_MODE,
                instructions="What was the outcome of the grasp in this frame?",
                options=[lb.Option(value=opt) for opt in FAILURE_OPTIONS],
            ),
        ],
    )
