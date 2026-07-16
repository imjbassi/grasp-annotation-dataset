"""Provision a Labelbox project + ontology for grasp annotation.

Idempotent-ish: if an ontology or project with the configured name already
exists it is reused instead of duplicated, so re-running is safe.

Usage
-----
    python -m grasp_annot.create_project
"""

from __future__ import annotations

import sys

import labelbox as lb

from .config import Settings
from .ontology import build_ontology


def _find_existing_ontology(client: lb.Client, name: str):
    for onto in client.get_ontologies(name):
        if onto.name == name:
            return onto
    return None


def _find_existing_project(client: lb.Client, name: str):
    for proj in client.get_projects(where=lb.Project.name == name):
        if proj.name == name:
            return proj
    return None


def create_project(settings: Settings) -> dict:
    """Create (or reuse) the ontology and project. Returns their ids."""
    client = lb.Client(api_key=settings.api_key)

    ontology = _find_existing_ontology(client, settings.ontology_name)
    if ontology is None:
        ontology = client.create_ontology(
            name=settings.ontology_name,
            normalized=build_ontology().asdict(),
            media_type=lb.MediaType.Image,
        )
        print(f"[create] ontology '{ontology.name}' ({ontology.uid})")
    else:
        print(f"[reuse]  ontology '{ontology.name}' ({ontology.uid})")

    project = _find_existing_project(client, settings.project_name)
    if project is None:
        project = client.create_project(
            name=settings.project_name,
            media_type=lb.MediaType.Image,
            description=(
                "Robot manipulation grasp annotation: grasp events (bbox), "
                "object contacts (keypoint), and per-frame failure modes."
            ),
        )
        project.connect_ontology(ontology)
        print(f"[create] project  '{project.name}' ({project.uid})")
    else:
        print(f"[reuse]  project  '{project.name}' ({project.uid})")

    print("\nOpen it in the app:")
    print(f"  https://app.labelbox.com/projects/{project.uid}")
    return {"project_id": project.uid, "ontology_id": ontology.uid}


def main() -> int:
    settings = Settings.from_env()
    create_project(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
