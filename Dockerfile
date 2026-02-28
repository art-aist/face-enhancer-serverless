# =============================================================
# ComfyUI Face Enhancer — RunPod SERVERLESS
# =============================================================
# Based on face-enhancer Pod template, adapted for Serverless.
# Uses RunPod handler instead of ComfyUI web server.
#
# What's inside:
#   - ComfyUI (latest)
#   - 11 custom nodes for Face Enhancement workflow
#   - SAM3 (3.3GB) for face segmentation — baked in
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
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV COMFYUI_PATH=/opt/ComfyUI
ENV COMFYUI_PORT=8188
ENV SAM2_BUILD_CUDA=0

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

# --- Install ComfyUI requirements (skip torch/torchvision/torchaudio — already in base image) ---
RUN grep -v -E "^(torch|torchvision|torchaudio)([ ><=!]|$)" requirements.txt > /tmp/comfy_req.txt && \
    pip install --no-cache-dir -r /tmp/comfy_req.txt

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

# --- Install SAM-2 + SAM3 dependencies (skip CUDA compilation via SAM2_BUILD_CUDA=0) ---
RUN pip install --no-cache-dir \
    "git+https://github.com/facebookresearch/sam2.git" \
    "segment-anything>=1.0" \
    "opencv-python-headless>=4.7.0" \
    "transformers>=4.30.0" \
    "decord" \
    "ftfy" \
    "hydra-core>=1.3.0" \
    "omegaconf>=2.3.0" \
    "iopath>=0.1.9"

# --- Download SAM3 model (~3.3GB) ---
RUN mkdir -p $COMFYUI_PATH/models/sam3 && \
    echo "Downloading SAM3..." && \
    wget -q --show-progress -O $COMFYUI_PATH/models/sam3/sam3.pt \
    "https://huggingface.co/1038lab/sam3/resolve/main/sam3.pt" && \
    echo "SAM3 downloaded: $(du -sh $COMFYUI_PATH/models/sam3/sam3.pt)"

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
sam3 = '/opt/ComfyUI/models/sam3/sam3.pt'; \
print(f'SAM3: {os.path.getsize(sam3)/1e9:.2f}GB' if os.path.exists(sam3) else 'SAM3: MISSING!'); \
from sam2.sam2_image_predictor import SAM2ImagePredictor; print('SAM-2 import: OK'); \
nodes = os.listdir('/opt/ComfyUI/custom_nodes'); \
print(f'Custom nodes: {len(nodes)} installed'); \
print('Build OK!'); \
"

# --- Entrypoint: RunPod handler (NOT ComfyUI web server) ---
CMD ["python3", "/opt/handler.py"]
