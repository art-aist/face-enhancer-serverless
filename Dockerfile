# =============================================================
# ComfyUI Face Enhancer — RunPod SERVERLESS
# =============================================================
# Based on face-enhancer Pod template, adapted for Serverless.
# Uses RunPod handler instead of ComfyUI web server.
#
# What's inside:
#   - ComfyUI (latest)
#   - 11 custom nodes for Face Enhancement workflow
#   - SAM ViT-L (1.25GB) + GroundingDINO (694MB) — baked in
#   - RunPod serverless handler
#   - DWPose: auto-downloaded on first run
#   - GeminiImage2Node: built into ComfyUI core
#
# Build on RunPod Pod (Docker-in-Docker):
#   bash build_and_push.sh <dockerhub_user>
# =============================================================

FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

LABEL maintainer="art@aist.digital"
LABEL description="ComfyUI Face Enhancer — RunPod Serverless, any GPU with CUDA 12+"

# --- Env ---
ENV DEBIAN_FRONTEND=noninteractive
ENV COMFYUI_PATH=/opt/ComfyUI
ENV COMFYUI_PORT=8188

# --- System deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Clone ComfyUI (pin to stable commit) ---
RUN git clone https://github.com/comfyanonymous/ComfyUI.git $COMFYUI_PATH

WORKDIR $COMFYUI_PATH

# --- Install ComfyUI requirements ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Install RunPod SDK ---
RUN pip install --no-cache-dir \
    "runpod>=1.7.0" \
    "requests>=2.31.0"

# --- Install custom nodes (same as face-enhancer Pod) ---
RUN cd $COMFYUI_PATH/custom_nodes && \
    git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git && \
    git clone --depth 1 https://github.com/1038lab/ComfyUI-RMBG.git && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git && \
    git clone --depth 1 https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git && \
    git clone --depth 1 https://github.com/yolain/ComfyUI-Easy-Use.git && \
    git clone --depth 1 https://github.com/rgthree/rgthree-comfy.git && \
    git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    git clone --depth 1 https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-WanVideoWrapper.git && \
    git clone --depth 1 https://github.com/cubiq/ComfyUI_essentials.git && \
    git clone --depth 1 https://github.com/melMass/comfy_mtb.git

# --- Install node pip requirements ---
RUN FAILED="" && \
    for d in $COMFYUI_PATH/custom_nodes/*/; do \
      name=$(basename "$d"); \
      if [ -f "$d/requirements.txt" ]; then \
        echo "=== pip: $name ==="; \
        pip install --no-cache-dir -r "$d/requirements.txt" 2>&1 || FAILED="$FAILED $name"; \
      fi; \
      if [ -f "$d/install.py" ]; then \
        echo "=== install.py: $name ==="; \
        (cd "$d" && python3 install.py) 2>&1 || echo "  [warn] install.py failed for $name"; \
      fi; \
    done && \
    if [ -n "$FAILED" ]; then \
      echo "WARNING: pip failed for:$FAILED (may be ok, check at runtime)"; \
    else \
      echo "All node deps installed OK"; \
    fi

# --- Download SAM ViT-L (~1.25GB) ---
RUN mkdir -p $COMFYUI_PATH/models/sams && \
    echo "Downloading SAM ViT-L..." && \
    wget -q --show-progress -O $COMFYUI_PATH/models/sams/sam_vit_l_0b3195.pth \
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth" && \
    echo "SAM ViT-L downloaded: $(du -sh $COMFYUI_PATH/models/sams/sam_vit_l_0b3195.pth)"

# --- Download GroundingDINO SwinT OGC (~694MB) ---
RUN mkdir -p $COMFYUI_PATH/models/grounding-dino && \
    echo "Downloading GroundingDINO..." && \
    wget -q --show-progress -O $COMFYUI_PATH/models/grounding-dino/groundingdino_swint_ogc.pth \
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" && \
    echo "GroundingDINO downloaded: $(du -sh $COMFYUI_PATH/models/grounding-dino/groundingdino_swint_ogc.pth)"

# --- Copy handler ---
COPY handler.py /opt/handler.py

# --- Final check ---
RUN python3 -c "\
import torch; \
print('=== Face Enhancer Serverless — Final Check ==='); \
print(f'PyTorch: {torch.__version__}'); \
print(f'CUDA: {torch.version.cuda}'); \
import runpod; print(f'RunPod SDK: {runpod.__version__}'); \
import numpy; print(f'NumPy: {numpy.__version__}'); \
import os; \
sams = '/opt/ComfyUI/models/sams/sam_vit_l_0b3195.pth'; \
dino = '/opt/ComfyUI/models/grounding-dino/groundingdino_swint_ogc.pth'; \
print(f'SAM ViT-L: {os.path.getsize(sams)/1e9:.2f}GB' if os.path.exists(sams) else 'SAM: MISSING!'); \
print(f'GroundingDINO: {os.path.getsize(dino)/1e9:.2f}GB' if os.path.exists(dino) else 'GroundingDINO: MISSING!'); \
nodes = os.listdir('/opt/ComfyUI/custom_nodes'); \
print(f'Custom nodes: {len(nodes)} installed'); \
print('Build OK!'); \
"

# --- Entrypoint: RunPod handler (NOT ComfyUI web server) ---
CMD ["python3", "/opt/handler.py"]
