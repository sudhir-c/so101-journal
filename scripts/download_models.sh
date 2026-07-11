#!/usr/bin/env bash
# Fetch the MediaPipe Tasks model bundles.
#
# The Tasks API (mediapipe >= 0.10) does NOT ship model weights inside the wheel
# -- unlike the old mp.solutions API, which bundled them. These .task files are
# required before the tracker will start.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p models
cd models

base="https://storage.googleapis.com/mediapipe-models"

fetch() {
  local url="$1" out="$2"
  if [[ -s "$out" ]]; then
    echo "  have  $out"
    return
  fi
  echo "  get   $out"
  curl -sSL --fail -o "$out" "$url"
}

echo "Downloading MediaPipe model bundles into models/ ..."
fetch "$base/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" pose_landmarker_full.task
fetch "$base/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" pose_landmarker_lite.task
fetch "$base/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"           hand_landmarker.task

echo
ls -lh ./*.task
echo
echo "Done."
