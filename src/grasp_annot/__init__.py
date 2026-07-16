"""grasp_annot — data-ops tooling for a robot-manipulation grasp annotation dataset.

Modules:
    config           Centralised, environment-driven configuration.
    ontology         Declarative annotation schema (grasp / contact / failure).
    create_project   Provision a Labelbox project + ontology from the schema.
    upload_data      Register extracted frames as Labelbox data rows + batch.
    export_coco      Pull completed labels and serialise them to COCO JSON.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
