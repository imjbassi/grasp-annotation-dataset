.PHONY: help install download frames project upload export test demo video clean

help:
	@echo "Robot grasp annotation — data-ops pipeline"
	@echo ""
	@echo "  make install    Install Python dependencies"
	@echo "  make download   Download the BridgeData V2 sample into data/raw"
	@echo "  make frames     Extract frames from data/raw into data/frames"
	@echo "  make project    Create the Labelbox project + ontology"
	@echo "  make upload     Upload data/frames to Labelbox and batch them"
	@echo "  make export     Export completed labels to COCO JSON"
	@echo "  make demo       Generate a tiny synthetic clip (no download needed)"
	@echo "  make video      Render an MP4 walkthrough of the pipeline"
	@echo "  make test       Run offline unit tests"
	@echo "  make clean      Remove generated data artifacts"

install:
	pip install -r requirements.txt

download:
	bash scripts/download_dataset.sh bridge

frames:
	python scripts/extract_frames.py --every 10

project:
	python -m grasp_annot.create_project

upload:
	python -m grasp_annot.upload_data

export:
	python -m grasp_annot.export_coco

demo:
	python scripts/make_demo_clip.py

video:
	python scripts/make_demo_video.py

test:
	PYTHONPATH=src pytest -q

clean:
	rm -rf data/raw/* data/frames/* data/exports/*
	find data -name '.gitkeep' -exec touch {} +
