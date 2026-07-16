#!/usr/bin/env bash
#
# download_dataset.sh — fetch a free, publicly available robot-arm manipulation
# dataset into data/raw/.
#
# Default source: BridgeData V2 (UC Berkeley RAIL Lab) — 60k+ teleoperated
# manipulation trajectories (pick, place, push, grasp) recorded from real WidowX
# robot arms. It is released for free research use and is distributed as plain
# RGB image sequences + mp4, so no robot, GPU, or simulator is needed.
#   Project page: https://rail.eecs.berkeley.edu/datasets/bridge_release/
#
# Alternative source: Open X-Embodiment "jaco_play" (Kinova Jaco arm doing
# tabletop pick-and-place), hosted on a public Google Cloud bucket. Enable it
# with:  ./download_dataset.sh jaco
#
# Both are large; we grab a single small shard so the whole pipeline is
# demonstrable on a laptop. Pass a second arg to change the destination.
#
set -euo pipefail

SOURCE="${1:-bridge}"
DEST="${2:-data/raw}"
mkdir -p "$DEST"

case "$SOURCE" in
  bridge)
    # A single scene tarball (~a few hundred MB of image sequences).
    # See the project page for the full list of scene archives.
    URL="https://rail.eecs.berkeley.edu/datasets/bridge_release/data/demos_8_17.zip"
    echo "[download] BridgeData V2 sample -> $DEST"
    echo "[download] $URL"
    curl -L --fail --retry 3 -o "$DEST/bridge_sample.zip" "$URL"
    echo "[download] unzipping ..."
    unzip -q -o "$DEST/bridge_sample.zip" -d "$DEST/bridge"
    echo "[download] done. Raw frames under $DEST/bridge"
    ;;

  jaco)
    # Requires gsutil (part of the Google Cloud SDK). Public, no auth needed.
    if ! command -v gsutil >/dev/null 2>&1; then
      echo "gsutil not found. Install the Google Cloud SDK:" >&2
      echo "  https://cloud.google.com/sdk/docs/install" >&2
      exit 1
    fi
    echo "[download] Open X-Embodiment jaco_play (first shard) -> $DEST/jaco_play"
    mkdir -p "$DEST/jaco_play"
    gsutil -m cp \
      "gs://gresearch/robotics/jaco_play/0.1.0/jaco_play-train.tfrecord-00000-of-00001" \
      "gs://gresearch/robotics/jaco_play/0.1.0/dataset_info.json" \
      "gs://gresearch/robotics/jaco_play/0.1.0/features.json" \
      "$DEST/jaco_play/"
    echo "[download] done. RLDS shards under $DEST/jaco_play"
    echo "[note] jaco_play is in RLDS/TFRecord format; extract_frames.py reads it"
    echo "       when tensorflow-datasets is installed (see requirements-extra)."
    ;;

  *)
    echo "Unknown source '$SOURCE'. Use 'bridge' or 'jaco'." >&2
    exit 1
    ;;
esac
