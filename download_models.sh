#!/usr/bin/env bash
# Fetch the model weights that are too large to keep in git.
#
# The weights live as a single archive attached to a GitHub Release. Set
# MODELS_URL to that asset's URL (or edit the default below), then run this
# script from the repo root:
#
#     MODELS_URL="https://github.com/<owner>/<repo>/releases/download/models-v1/models.tar.gz" \
#       bash download_models.sh
#
# The YoutuReID embedder is NOT included here — the siglip service downloads it
# automatically from Hugging Face on first boot.
set -euo pipefail
cd "$(dirname "$0")"

MODELS_URL="${MODELS_URL:-}"

need=(
  "ai/models/rtdetr_r50_uint8.onnx"
  "ai/models/face_detection_yunet_2023mar.onnx"
  "face/models/recognition.onnx"
  "face/models/face_landmarker.task"
  "plate/models/yolov9-t-640-plate-end2end.onnx"
  "plate/models/ocr/inference.pdiparams"
)

have_all=true
for f in "${need[@]}"; do [ -f "$f" ] || have_all=false; done
if $have_all; then echo "✓ all model weights already present."; exit 0; fi

if [ -z "$MODELS_URL" ]; then
  cat <<EOF
Model weights are missing. Set MODELS_URL to the release archive and re-run:

  MODELS_URL="https://github.com/<owner>/<repo>/releases/download/models-v1/models.tar.gz" \\
    bash download_models.sh

Expected files (place under the repo if you have them another way):
$(printf '  %s\n' "${need[@]}")

(The YoutuReID embedder auto-downloads on first boot — no action needed.)
EOF
  exit 1
fi

echo "Downloading model bundle from: $MODELS_URL"
tmp="$(mktemp)"
curl -fL --progress-bar "$MODELS_URL" -o "$tmp"
echo "Extracting…"
tar -xzf "$tmp" -C .
rm -f "$tmp"
echo "✓ model weights installed."
