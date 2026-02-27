#!/bin/bash
# =============================================================
# Face Enhancer SERVERLESS — Docker Build & Push
# =============================================================
# Usage: bash build_and_push.sh [dockerhub_user]
# Example: bash build_and_push.sh art1aist
#
# Run this ON A RUNPOD POD (with Docker access), NOT on local Mac.
# Docker is available on RunPod Pods by default.
# =============================================================
set -e

DOCKERHUB_USER="${1:-art1aist}"
IMAGE_NAME="comfyui-face-enhancer-serverless"
TAG="latest"
FULL_IMAGE="$DOCKERHUB_USER/$IMAGE_NAME:$TAG"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "=================================================="
echo "  Face Enhancer SERVERLESS — Docker Build"
echo "  Image: $FULL_IMAGE"
echo "  Base:  pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
echo "  Mode:  RunPod Serverless (NOT Pod)"
echo "=================================================="
echo ""

# Check we have handler.py and Dockerfile
if [ ! -f "$SCRIPT_DIR/Dockerfile" ] || [ ! -f "$SCRIPT_DIR/handler.py" ]; then
    echo "ERROR: Run this script from the serverless/ directory"
    echo "  cd /path/to/upscaler_banana_pro/serverless"
    echo "  bash build_and_push.sh $DOCKERHUB_USER"
    exit 1
fi

# ===== STEP 1: Docker Hub Login =====
echo "[1/4] Docker Hub Login..."
echo ""
echo "Нужен Access Token (не пароль!):"
echo "  hub.docker.com → Account Settings → Security → New Access Token"
echo ""
docker login -u "$DOCKERHUB_USER"
echo ""

# ===== STEP 2: Build =====
echo "[2/4] Building Docker image..."
echo "  Это займёт 20-40 минут (скачивание SAM 1.25GB + DINO 694MB + nodes)"
echo ""

docker build \
    -f "$SCRIPT_DIR/Dockerfile" \
    -t "$FULL_IMAGE" \
    --progress=plain \
    "$SCRIPT_DIR"

echo ""
echo "Build complete! Checking image size..."
docker image inspect "$FULL_IMAGE" --format='{{.Size}}' | \
    python3 -c "import sys; s=int(sys.stdin.read()); print(f'  Image size: {s/1e9:.1f} GB')"
echo ""

# ===== STEP 3: Quick Verify =====
echo "[3/4] Quick verification (no GPU needed)..."
docker run --rm "$FULL_IMAGE" python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available in image: {torch.version.cuda}')
import runpod
print(f'RunPod SDK: {runpod.__version__}')
import os
sams = '/opt/ComfyUI/models/sams/sam_vit_l_0b3195.pth'
dino = '/opt/ComfyUI/models/grounding-dino/groundingdino_swint_ogc.pth'
print(f'SAM ViT-L: {os.path.getsize(sams)/1e9:.2f}GB' if os.path.exists(sams) else 'SAM: MISSING!')
print(f'GroundingDINO: {os.path.getsize(dino)/1e9:.2f}GB' if os.path.exists(dino) else 'GroundingDINO: MISSING!')
nodes = os.listdir('/opt/ComfyUI/custom_nodes')
print(f'Custom nodes ({len(nodes)}): {sorted(nodes)}')
print('Verification PASSED!')
"
echo ""

# ===== STEP 4: Push =====
echo "[4/4] Pushing to Docker Hub..."
docker push "$FULL_IMAGE"

echo ""
echo "=================================================="
echo "  DONE! Image pushed: $FULL_IMAGE"
echo ""
echo "  Следующий шаг — создай Serverless Endpoint на RunPod:"
echo ""
echo "  1. Открой: https://www.runpod.io/console/serverless"
echo "  2. Нажми: + New Endpoint"
echo "  3. Настройки:"
echo "  ┌─────────────────────────────────────────────────┐"
echo "  │  Name: face-enhancer"
echo "  │  Container Image: $FULL_IMAGE"
echo "  │  GPU: L4 24GB (или любой с 8GB+ VRAM)"
echo "  │  Workers:"
echo "  │    Min: 0 (Flex, scale to zero)"
echo "  │    Max: 1"
echo "  │  Idle Timeout: 30 sec"
echo "  │  Execution Timeout: 600 sec"
echo "  │  Container Disk: 20 GB"
echo "  │"
echo "  │  Environment Variables:"
echo "  │    GEMINI_API_KEY = your_banana_pro_api_key"
echo "  │    COMFYUI_STARTUP_TIMEOUT = 120"
echo "  └─────────────────────────────────────────────────┘"
echo ""
echo "  4. Нажми: Create Endpoint"
echo "  5. Скопируй Endpoint ID (нужен для Gradio UI)"
echo "  6. API Key: Account → Settings → API Keys"
echo "=================================================="
