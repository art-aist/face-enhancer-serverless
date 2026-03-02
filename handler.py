"""
RunPod Serverless Handler for ComfyUI Face Enhancer.

Supports three modes:
  - "detect":  Run SAM3 → return mask preview overlay
  - "enhance": Run SAM3 → Python square crop → Gemini → return enhanced + mask + bbox
  - "workflow": Legacy mode — run arbitrary workflow JSON

Input format:
{
    "input": {
        "mode": "detect" | "enhance" | "workflow",

        # For detect/enhance:
        "image": "<base64>",
        "image_name": "photo.png",
        "segment_pick": 1,

        # For enhance only:
        "model": "Nano Banana 2 (Gemini 3.1 Flash Image)",
        "resolution": "2K",
        "prompt": "...",

        # For legacy workflow mode:
        "workflow": { ... },
        "images": [{ "name": "...", "image": "<base64>" }]
    }
}
"""

import os
import sys
import json
import copy
import time
import uuid
import random
import base64
import subprocess
import threading
import requests
import numpy as np
from io import BytesIO

from PIL import Image
import runpod

# --- Configuration ---
COMFYUI_PATH = os.environ.get("COMFYUI_PATH", "/workspace/ComfyUI")
COMFYUI_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
COMFYUI_HOST = f"http://127.0.0.1:{COMFYUI_PORT}"
COMFYUI_STARTUP_TIMEOUT = int(os.environ.get("COMFYUI_STARTUP_TIMEOUT", "120"))
COMFYUI_ARGS = os.environ.get("COMFYUI_ARGS", "")

# --- ComfyUI API auth (Firebase) ---
FIREBASE_API_KEY = os.environ.get("FIREBASE_API_KEY", "")
COMFY_REFRESH_TOKEN = os.environ.get("COMFY_REFRESH_TOKEN", "")

comfyui_process = None
_cached_token = {"access_token": None, "expires_at": 0}

# Max input image dimension
MAX_IMAGE_DIM = 5000

# --- Embedded Workflows ---

DETECT_WORKFLOW = {
    "1": {
        "inputs": {"image": "input.png"},
        "class_type": "LoadImage",
        "_meta": {"title": "Load Image"},
    },
    "2": {
        "inputs": {
            "prompt": "head",
            "output_mode": "Merged",
            "confidence_threshold": 0.25,
            "max_segments": 5,
            "segment_pick": 1,
            "mask_blur": 4,
            "mask_offset": 0,
            "device": "Auto",
            "invert_output": False,
            "unload_model": False,
            "background": "Color",
            "background_color": "#222222",
            "image": ["1", 0],
        },
        "class_type": "SAM3Segment",
        "_meta": {"title": "SAM3 Segmentation (RMBG)"},
    },
    "3": {
        "inputs": {
            "mask_opacity": 0.8,
            "mask_color": "255, 0, 0",
            "pass_through": True,
            "image": ["1", 0],
            "mask": ["2", 1],
        },
        "class_type": "ImageAndMaskPreview",
        "_meta": {"title": "ImageAndMaskPreview"},
    },
    "4": {
        "inputs": {"images": ["3", 0]},
        "class_type": "PreviewImage",
        "_meta": {"title": "Preview Image"},
    },
}

SAM3_BBOX_WORKFLOW = {
    "1": {
        "inputs": {"image": "input.png"},
        "class_type": "LoadImage",
        "_meta": {"title": "Load Image"},
    },
    "2": {
        "inputs": {
            "prompt": "head",
            "output_mode": "Merged",
            "confidence_threshold": 0.25,
            "max_segments": 5,
            "segment_pick": 1,
            "mask_blur": 0,
            "mask_offset": 0,
            "device": "Auto",
            "invert_output": False,
            "unload_model": False,
            "background": "Color",
            "background_color": "#222222",
            "image": ["1", 0],
        },
        "class_type": "SAM3Segment",
        "_meta": {"title": "SAM3 Segmentation (RMBG)"},
    },
    "5": {
        "inputs": {"mask": ["2", 1]},
        "class_type": "MaskPreview",
        "_meta": {"title": "Mask Preview"},
    },
    "7": {
        "inputs": {"invert": False, "mask": ["2", 1]},
        "class_type": "Bbox From Mask (mtb)",
        "_meta": {"title": "Bbox From Mask (mtb)"},
    },
    "9": {
        "inputs": {"text": "", "anything": ["7", 0]},
        "class_type": "easy showAnything",
        "_meta": {"title": "Show Any"},
    },
}

GEMINI_WORKFLOW = {
    "14": {
        "inputs": {"image": "square_crop.png"},
        "class_type": "LoadImage",
        "_meta": {"title": "Load Image"},
    },
    "15": {
        "inputs": {
            "prompt": "",
            "model": "Nano Banana 2 (Gemini 3.1 Flash Image)",
            "seed": 0,
            "aspect_ratio": "1:1",
            "resolution": "2K",
            "response_modalities": "IMAGE",
            "system_prompt": (
                "You are an expert image-generation engine. You must ALWAYS produce an image.\n"
                "Interpret all user input—regardless of format, intent, or abstraction—"
                "as literal visual directives for image composition.\n"
                "If a prompt is conversational or lacks specific visual details, you must "
                "creatively invent a concrete visual scenario that depicts the concept.\n"
                "Prioritize generating the visual representation above any text, formatting, "
                "or conversational requests."
            ),
            "images": ["14", 0],
        },
        "class_type": "GeminiImage2Node",
        "_meta": {"title": "Nano Banana Pro (Google Gemini Image)"},
    },
    "16": {
        "inputs": {"images": ["15", 0]},
        "class_type": "PreviewImage",
        "_meta": {"title": "Preview Image"},
    },
}

DEFAULT_PROMPT = (
    "Preserve the exact composition, camera angle, and framing of the first original image. "
    "Keep all main elements in their current positions, maintaining the same perspective "
    "and horizon line. hyper-detailed skin with subsurface scattering effect, micro skin "
    "texture with soft peach fuzz, nearly invisible pores and natural expression lines, "
    "slightly oily T-zone with natural shine on forehead and cheeks, moist inner corners "
    "of the eyes, realistic eyelashes with varied lengths, lifelike highlights in the eyes, "
    "soft glossy effect on the lips, natural lip texture, baby hairs along the hairline, "
    "detailed hair strands with slight messiness, realistic volume and natural color transitions"
)


# ============================================================
# ComfyUI Management
# ============================================================

def get_comfy_auth_token():
    """Get a fresh ComfyUI auth token using Firebase refresh token."""
    if not FIREBASE_API_KEY or not COMFY_REFRESH_TOKEN:
        print("[handler] No Firebase credentials — skipping auth")
        return None

    now = time.time()
    if _cached_token["access_token"] and now < _cached_token["expires_at"] - 60:
        return _cached_token["access_token"]

    resp = requests.post(
        f"https://securetoken.googleapis.com/v1/token?key={FIREBASE_API_KEY}",
        data={
            "grant_type": "refresh_token",
            "refresh_token": COMFY_REFRESH_TOKEN,
        },
        timeout=10,
    )

    if resp.status_code == 200:
        data = resp.json()
        _cached_token["access_token"] = data["id_token"]
        _cached_token["expires_at"] = now + int(data.get("expires_in", 3600))
        print("[handler] Auth token refreshed OK")
        return _cached_token["access_token"]
    else:
        print(f"[handler] Token refresh failed ({resp.status_code}): {resp.text}")
        return None


def start_comfyui():
    """Start ComfyUI server in background."""
    global comfyui_process

    cmd = [
        sys.executable, "main.py",
        "--listen", "127.0.0.1",
        "--port", str(COMFYUI_PORT),
        "--disable-auto-launch",
    ]
    if COMFYUI_ARGS:
        cmd.extend(COMFYUI_ARGS.split())

    print(f"[handler] Starting ComfyUI: {' '.join(cmd)}")
    comfyui_process = subprocess.Popen(
        cmd,
        cwd=COMFYUI_PATH,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    def log_output():
        for line in comfyui_process.stdout:
            print(f"[comfyui] {line}", end="")

    log_thread = threading.Thread(target=log_output, daemon=True)
    log_thread.start()


def wait_for_comfyui():
    """Wait until ComfyUI is ready to accept requests."""
    print(f"[handler] Waiting for ComfyUI (timeout: {COMFYUI_STARTUP_TIMEOUT}s)...")
    start = time.time()

    while time.time() - start < COMFYUI_STARTUP_TIMEOUT:
        try:
            resp = requests.get(f"{COMFYUI_HOST}/system_stats", timeout=2)
            if resp.status_code == 200:
                elapsed = time.time() - start
                print(f"[handler] ComfyUI ready in {elapsed:.1f}s")
                return True
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            print(f"[handler] Health check error: {e}")

        if comfyui_process and comfyui_process.poll() is not None:
            print(f"[handler] ComfyUI crashed with code {comfyui_process.returncode}")
            return False

        time.sleep(2)

    print(f"[handler] ComfyUI startup timeout ({COMFYUI_STARTUP_TIMEOUT}s)")
    return False


# ============================================================
# ComfyUI API Functions
# ============================================================

def upload_image(name, image_base64):
    """Upload a base64 image to ComfyUI's input directory."""
    image_data = base64.b64decode(image_base64)

    resp = requests.post(
        f"{COMFYUI_HOST}/upload/image",
        files={"image": (name, BytesIO(image_data), "image/png")},
        data={"overwrite": "true"},
    )

    if resp.status_code == 200:
        result = resp.json()
        print(f"[handler] Uploaded: {name} -> {result.get('name', name)}")
        return result.get("name", name)
    else:
        raise RuntimeError(f"Image upload failed ({resp.status_code}): {resp.text}")


def upload_pil_image(name, pil_image):
    """Upload a PIL Image object to ComfyUI's input directory."""
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)

    resp = requests.post(
        f"{COMFYUI_HOST}/upload/image",
        files={"image": (name, buf, "image/png")},
        data={"overwrite": "true"},
    )

    if resp.status_code == 200:
        result = resp.json()
        print(f"[handler] Uploaded PIL: {name} -> {result.get('name', name)}")
        return result.get("name", name)
    else:
        raise RuntimeError(f"PIL upload failed ({resp.status_code}): {resp.text}")


def queue_workflow(workflow):
    """Send workflow to ComfyUI's /prompt endpoint."""
    client_id = str(uuid.uuid4())

    payload = {
        "prompt": workflow,
        "client_id": client_id,
    }

    auth_token = get_comfy_auth_token()
    if auth_token:
        payload["extra_data"] = {
            "auth_token_comfy_org": auth_token,
        }

    resp = requests.post(
        f"{COMFYUI_HOST}/prompt",
        json=payload,
    )

    if resp.status_code == 200:
        result = resp.json()
        prompt_id = result.get("prompt_id")
        print(f"[handler] Queued workflow: prompt_id={prompt_id}")
        return prompt_id, client_id
    else:
        raise RuntimeError(f"Workflow queue failed ({resp.status_code}): {resp.text}")


def wait_for_completion(prompt_id, timeout=600, poll_interval=2):
    """Poll ComfyUI until the workflow completes or fails."""
    start = time.time()

    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{COMFYUI_HOST}/history/{prompt_id}", timeout=5)
            if resp.status_code == 200:
                history = resp.json()
                if prompt_id in history:
                    entry = history[prompt_id]
                    status = entry.get("status", {})
                    if status.get("completed", False):
                        elapsed = time.time() - start
                        print(f"[handler] Workflow completed in {elapsed:.1f}s")
                        return entry
                    if status.get("status_str") == "error":
                        messages = status.get("messages", [])
                        raise RuntimeError(f"Workflow error: {messages}")
        except requests.exceptions.ConnectionError:
            pass
        except RuntimeError:
            raise
        except Exception as e:
            print(f"[handler] Poll error: {e}")

        time.sleep(poll_interval)

    raise RuntimeError(f"Workflow timeout after {timeout}s")


def fetch_image_from_history(history_entry, node_id):
    """Download an image from a specific node in ComfyUI history."""
    outputs = history_entry.get("outputs", {})
    node_output = outputs.get(node_id, {})

    if "images" not in node_output or not node_output["images"]:
        raise RuntimeError(f"No images in node {node_id} output")

    img_info = node_output["images"][0]
    resp = requests.get(
        f"{COMFYUI_HOST}/view",
        params={
            "filename": img_info["filename"],
            "subfolder": img_info.get("subfolder", ""),
            "type": img_info.get("type", "output"),
        },
        timeout=30,
    )

    if resp.status_code == 200:
        print(f"[handler] Fetched node {node_id}: {img_info['filename']} "
              f"({len(resp.content) // 1024}KB)")
        return resp.content
    else:
        raise RuntimeError(f"Failed to fetch node {node_id}: {resp.status_code}")


def extract_text_from_history(history_entry, node_id):
    """Extract text output from a display node (e.g. easy showAnything)."""
    outputs = history_entry.get("outputs", {})
    node_output = outputs.get(node_id, {})

    # easy showAnything stores text in "text" key
    if "text" in node_output:
        text_list = node_output["text"]
        if isinstance(text_list, list) and text_list:
            return text_list[0]
        return str(text_list)

    print(f"[handler] Warning: no text output in node {node_id}. Keys: {list(node_output.keys())}")
    return None


# ============================================================
# Python Image Processing
# ============================================================

def make_square_crop(pil_image, bbox, min_size=512, padding=32):
    """Create a square crop centered on the mask bbox.

    Args:
        pil_image: PIL Image (original full image)
        bbox: [x, y, width, height] from BboxFromMask
        min_size: minimum crop side length
        padding: pixels to add around bbox

    Returns:
        (cropped_pil_image, crop_info) where crop_info = {x, y, size}
    """
    img_w, img_h = pil_image.size
    bx, by, bw, bh = bbox

    # Square size: max of bbox dims + padding, at least min_size
    s = max(bw, bh) + padding * 2
    s = max(s, min_size)

    # Center on bbox center
    cx = bx + bw // 2
    cy = by + bh // 2

    # Check if we need to pad the image
    if s > img_w or s > img_h:
        # Pad image to fit the square crop
        new_w = max(s, img_w)
        new_h = max(s, img_h)
        pad_x = (new_w - img_w) // 2
        pad_y = (new_h - img_h) // 2

        # Create padded image with edge-reflect
        padded = Image.new("RGB", (new_w, new_h))

        # Paste original in center
        padded.paste(pil_image, (pad_x, pad_y))

        # Fill padding with mirrored edges
        img_arr = np.array(pil_image)
        padded_arr = np.array(padded)

        # Top padding
        if pad_y > 0:
            flip_h = min(pad_y, img_h)
            padded_arr[pad_y - flip_h:pad_y, pad_x:pad_x + img_w] = img_arr[:flip_h][::-1]
        # Bottom padding
        bottom_pad = new_h - (pad_y + img_h)
        if bottom_pad > 0:
            flip_h = min(bottom_pad, img_h)
            padded_arr[pad_y + img_h:pad_y + img_h + flip_h, pad_x:pad_x + img_w] = img_arr[-flip_h:][::-1]
        # Left padding
        if pad_x > 0:
            flip_w = min(pad_x, img_w)
            padded_arr[pad_y:pad_y + img_h, pad_x - flip_w:pad_x] = img_arr[:, :flip_w][:, ::-1]
        # Right padding
        right_pad = new_w - (pad_x + img_w)
        if right_pad > 0:
            flip_w = min(right_pad, img_w)
            padded_arr[pad_y:pad_y + img_h, pad_x + img_w:pad_x + img_w + flip_w] = img_arr[:, -flip_w:][:, ::-1]

        pil_image = Image.fromarray(padded_arr)
        cx += pad_x
        cy += pad_y
        img_w, img_h = new_w, new_h
        print(f"[handler] Padded image to {new_w}x{new_h} (pad_x={pad_x}, pad_y={pad_y})")

    # Clamp crop position to image bounds
    crop_x = max(0, min(cx - s // 2, img_w - s))
    crop_y = max(0, min(cy - s // 2, img_h - s))

    # Ensure s doesn't exceed image dimensions (safety)
    s = min(s, img_w, img_h)

    cropped = pil_image.crop((crop_x, crop_y, crop_x + s, crop_y + s))

    print(f"[handler] Square crop: bbox=({bx},{by},{bw},{bh}), "
          f"crop=({crop_x},{crop_y},{s}x{s}), img=({img_w}x{img_h})")

    return cropped, {"x": crop_x, "y": crop_y, "size": s}


def _compress_to_jpeg(raw_bytes, max_size=1024, quality=85):
    """Compress image bytes to JPEG, optionally resizing if larger than max_size."""
    img = Image.open(BytesIO(raw_bytes))
    if img.mode == "RGBA":
        img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _image_to_b64(raw_bytes, compress=False):
    """Convert raw image bytes to base64 string, optionally compressing."""
    if compress:
        compressed = _compress_to_jpeg(raw_bytes)
        return base64.b64encode(compressed).decode("utf-8")
    return base64.b64encode(raw_bytes).decode("utf-8")


# ============================================================
# Mode Handlers
# ============================================================

def handle_detect(job_input):
    """Detect mode: run SAM3 → return mask preview overlay."""
    image_b64 = job_input.get("image", "")
    image_name = job_input.get("image_name", "input.png")
    segment_pick = job_input.get("segment_pick", 1)

    if not image_b64:
        return {"error": "Missing 'image' in input"}

    # Upload image
    uploaded_name = upload_image(image_name, image_b64)

    # Build detect workflow
    wf = copy.deepcopy(DETECT_WORKFLOW)
    wf["1"]["inputs"]["image"] = uploaded_name
    wf["2"]["inputs"]["segment_pick"] = segment_pick

    # Run workflow
    prompt_id, _ = queue_workflow(wf)
    history = wait_for_completion(prompt_id, timeout=120)

    # Fetch mask preview from node 4 (keep as PNG — JPEG destroys semi-transparent overlays)
    preview_bytes = fetch_image_from_history(history, "4")
    preview_b64 = _image_to_b64(preview_bytes, compress=False)

    return {
        "status": "success",
        "mode": "detect",
        "images": [{
            "node_id": "4",
            "image": preview_b64,
            "type": "mask_preview",
        }],
    }


def handle_enhance(job_input):
    """Enhance mode: SAM3 → Python square crop → Gemini → return results."""
    image_b64 = job_input.get("image", "")
    image_name = job_input.get("image_name", "input.png")
    segment_pick = job_input.get("segment_pick", 1)
    model = job_input.get("model", "Nano Banana 2 (Gemini 3.1 Flash Image)")
    resolution = job_input.get("resolution", "2K")
    prompt = job_input.get("prompt", DEFAULT_PROMPT)

    if not image_b64:
        return {"error": "Missing 'image' in input"}

    # Decode original image
    original_bytes = base64.b64decode(image_b64)
    original_image = Image.open(BytesIO(original_bytes))
    if original_image.mode == "RGBA":
        original_image = original_image.convert("RGB")

    img_w, img_h = original_image.size
    print(f"[handler] Original image: {img_w}x{img_h}")

    # Validate image size
    if max(img_w, img_h) > MAX_IMAGE_DIM:
        return {"error": f"Image too large ({img_w}x{img_h}). Max dimension: {MAX_IMAGE_DIM}px"}

    # === Step 1: Run SAM3 + BboxFromMask ===
    print("[handler] Step 1: Running SAM3...")
    uploaded_name = upload_image(image_name, image_b64)

    wf_sam3 = copy.deepcopy(SAM3_BBOX_WORKFLOW)
    wf_sam3["1"]["inputs"]["image"] = uploaded_name
    wf_sam3["2"]["inputs"]["segment_pick"] = segment_pick

    prompt_id, _ = queue_workflow(wf_sam3)
    history_sam3 = wait_for_completion(prompt_id, timeout=120)

    # Get mask image (node 5: PreviewImage of MASK_IMAGE) — keep as PNG for clean compositing
    mask_bytes = fetch_image_from_history(history_sam3, "5")
    mask_b64 = _image_to_b64(mask_bytes, compress=False)
    print(f"[handler] Mask image: {len(mask_bytes) // 1024}KB")

    # Get bbox text (node 9: Show Any)
    bbox_text = extract_text_from_history(history_sam3, "9")
    if not bbox_text:
        return {"error": "Failed to extract bbox from SAM3 output"}

    try:
        bbox = json.loads(bbox_text)
        print(f"[handler] Bbox: {bbox}")
    except (json.JSONDecodeError, TypeError) as e:
        return {"error": f"Failed to parse bbox: {bbox_text} ({e})"}

    if len(bbox) != 4:
        return {"error": f"Invalid bbox format: {bbox}"}

    # === Step 2: Python square crop ===
    print("[handler] Step 2: Computing square crop...")
    square_crop, crop_info = make_square_crop(original_image, bbox)
    crop_size = square_crop.size[0]
    print(f"[handler] Square crop: {crop_size}x{crop_size}")

    # Upscale to 1024 if smaller
    if crop_size < 1024:
        square_crop = square_crop.resize((1024, 1024), Image.LANCZOS)
        print(f"[handler] Upscaled crop to 1024x1024")

    # === Step 3: Run Gemini ===
    print("[handler] Step 3: Running Gemini...")
    crop_name = f"crop_{uuid.uuid4().hex[:8]}.png"
    uploaded_crop = upload_pil_image(crop_name, square_crop)

    wf_gemini = copy.deepcopy(GEMINI_WORKFLOW)
    wf_gemini["14"]["inputs"]["image"] = uploaded_crop
    wf_gemini["15"]["inputs"]["prompt"] = prompt
    wf_gemini["15"]["inputs"]["model"] = model
    wf_gemini["15"]["inputs"]["resolution"] = resolution
    wf_gemini["15"]["inputs"]["seed"] = random.randint(1, 2**53)

    prompt_id, _ = queue_workflow(wf_gemini)
    history_gemini = wait_for_completion(prompt_id, timeout=300)

    # Get enhanced image (node 16: PreviewImage)
    # Only compress if over 10MB (RunPod response limit ~20MB, leave room for mask + metadata)
    # 2K images (~5-8MB PNG) stay lossless; 4K images (25-30MB PNG) get JPEG compressed
    enhanced_bytes = fetch_image_from_history(history_gemini, "16")
    raw_size_kb = len(enhanced_bytes) // 1024
    print(f"[handler] Enhanced image raw: {raw_size_kb}KB")
    if len(enhanced_bytes) > 10 * 1024 * 1024:
        enhanced_bytes = _compress_to_jpeg(enhanced_bytes, max_size=4096, quality=95)
        print(f"[handler] Compressed to JPEG: {len(enhanced_bytes) // 1024}KB")
    else:
        print(f"[handler] Keeping as lossless PNG ({raw_size_kb}KB < 10MB threshold)")
    enhanced_b64 = base64.b64encode(enhanced_bytes).decode("utf-8")

    # === Return results ===
    return {
        "status": "success",
        "mode": "enhance",
        "bbox": bbox,
        "crop_info": crop_info,
        "images": [
            {
                "node_id": "16",
                "image": enhanced_b64,
                "type": "enhanced",
            },
            {
                "node_id": "5",
                "image": mask_b64,
                "type": "mask",
            },
        ],
    }


def handle_legacy_workflow(job_input):
    """Legacy mode: run arbitrary workflow JSON (backward compatible)."""
    workflow = job_input.get("workflow")
    if not workflow:
        return {"error": "Missing 'workflow' in input"}

    input_images = job_input.get("images", [])

    # Upload input images
    uploaded_names = {}
    for img in input_images:
        name = img.get("name", f"input_{uuid.uuid4().hex[:8]}.png")
        image_data = img.get("image", "")
        if image_data:
            uploaded_name = upload_image(name, image_data)
            uploaded_names[name] = uploaded_name

    # Patch workflow with uploaded image names
    if uploaded_names:
        for node_id, node in workflow.items():
            if node.get("class_type") == "LoadImage":
                inputs = node.get("inputs", {})
                original_name = inputs.get("image", "")
                if original_name in uploaded_names:
                    inputs["image"] = uploaded_names[original_name]
                elif len(uploaded_names) == 1:
                    inputs["image"] = list(uploaded_names.values())[0]

    # Queue workflow
    prompt_id, client_id = queue_workflow(workflow)

    # Wait for completion
    execution_timeout = job_input.get("timeout", 600)
    history_entry = wait_for_completion(prompt_id, timeout=execution_timeout)

    # Collect outputs
    output_nodes = job_input.get("output_nodes")
    only_nodes = set(output_nodes) if output_nodes else None
    compress_nodes = (only_nodes - {"68"}) if only_nodes else set()

    outputs = history_entry.get("outputs", {})
    images = []
    for node_id, node_output in outputs.items():
        if only_nodes and node_id not in only_nodes:
            continue
        if "images" in node_output:
            for img_info in node_output["images"]:
                resp = requests.get(
                    f"{COMFYUI_HOST}/view",
                    params={
                        "filename": img_info["filename"],
                        "subfolder": img_info.get("subfolder", ""),
                        "type": img_info.get("type", "output"),
                    },
                    timeout=30,
                )
                if resp.status_code == 200:
                    raw = resp.content
                    img_b64 = _image_to_b64(raw, compress=(node_id in compress_nodes))
                    images.append({
                        "filename": img_info["filename"],
                        "node_id": node_id,
                        "image": img_b64,
                    })

    if not images:
        return {"error": "Workflow completed but no output images found"}

    return {"status": "success", "images": images}


# ============================================================
# Main Handler
# ============================================================

def handler(job):
    """RunPod serverless handler — main entry point."""
    print("[handler] === Job received ===")
    job_input = job.get("input", {})
    mode = job_input.get("mode", "workflow")
    print(f"[handler] Mode: {mode}, keys: {list(job_input.keys())}")

    try:
        if mode == "detect":
            return handle_detect(job_input)
        elif mode == "enhance":
            return handle_enhance(job_input)
        else:
            return handle_legacy_workflow(job_input)
    except Exception as e:
        print(f"[handler] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ============================================================
# Startup
# ============================================================

if __name__ == "__main__":
    workspace_comfyui = "/workspace/ComfyUI"
    docker_comfyui = "/opt/ComfyUI"

    if not os.path.exists(workspace_comfyui) and os.path.exists(docker_comfyui):
        print("[handler] First run — copying ComfyUI to /workspace...")
        os.system(f"cp -r {docker_comfyui} {workspace_comfyui}")

    if os.path.exists(workspace_comfyui):
        os.environ["COMFYUI_PATH"] = workspace_comfyui
        COMFYUI_PATH = workspace_comfyui

    start_comfyui()

    if not wait_for_comfyui():
        print("[handler] FATAL: ComfyUI failed to start")
        sys.exit(1)

    print("[handler] Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
