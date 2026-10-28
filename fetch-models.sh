#!/usr/bin/env bash
# Pull the model weights this project needs. They are not stored in git.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models

get() {
  if [ -f "$2" ]; then
    echo "have $2"
  else
    echo "fetching $2"
    curl -fL --retry 2 -o "$2" "$1"
  fi
}

get https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task models/hand_landmarker.task
get https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task models/face_landmarker.task
get https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt yolo11n.pt

echo "done"
