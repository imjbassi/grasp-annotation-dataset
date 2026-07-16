# Annotator Guide — Grasp Annotation Schema

This is the reference annotators use inside Labelbox. It mirrors the machine
schema defined in [`../src/grasp_annot/ontology.py`](../src/grasp_annot/ontology.py).

## Tools

### 1. `grasp_event` — bounding box
Draw a **tight** box around the region where the gripper contacts (or attempts
to contact) the object, at the frame where the grasp is attempted. One box per
grasp attempt visible in the frame.

- Include both gripper fingertips and the contacted part of the object.
- Do **not** box the whole arm or the whole object.

### 2. `object_contact` — keypoint
Place a **single point** at the location where a fingertip first makes contact
with the object surface. If no contact is visible in the frame, omit it.

### 3. `failure_mode` — single-select classification
Choose exactly **one** outcome for the frame:

| Option      | Use when… |
| ----------- | --------- |
| `success`   | Object grasped and held securely. |
| `drop`      | Object was grasped but then slipped/fell out of the gripper. |
| `miss`      | Gripper closed but never secured the object. |
| `collision` | Gripper/arm struck the object or environment unintentionally. |
| `timeout`   | Attempt ran out of time without a clear success or failure. |

## Conventions

- One frame = one Labelbox data row. Label only what is visible **in that frame**.
- When ambiguous between `miss` and `drop`: if the object was ever enclosed by
  the fingers, it's `drop`; otherwise `miss`.
- Keypoint and bounding box are independent — a frame may have one, both, or
  neither, but `failure_mode` should always be set.
