#!/usr/bin/env bash
# Download the MediaPipe Hand Landmarker task bundle.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/models/hand_landmarker.task"
URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

mkdir -p "${ROOT}/models"
if [[ -f "${DEST}" ]]; then
  echo "Already present: ${DEST}"
  exit 0
fi

echo "Downloading Hand Landmarker model..."
curl -L --fail --output "${DEST}" "${URL}"
echo "Saved ${DEST}"
