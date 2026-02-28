"""
RunPod Serverless Handler for ComfyUI Face Enhancer.

Starts ComfyUI internally, accepts workflow + images via RunPod API,
runs the workflow, returns output images as base64.

Input format:
{
    "input": {
        "workflow": { ... },        # ComfyUI API-format workflow JSON
        "images": [                  # Optional: input images to upload
            {
                "name": "input.png",
                "image": "<base64>"
            }
        ]
    }
}

Output format:
{
    "output": {
        "images": ["<base64>", ...],  # Output images as base64
        "status": "success"
    }
}
"""

import os
import sys
import json
import time
import uuid
import base64
import signal
import subprocess
import threading
import requests
from io import BytesIO

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

# Cache for refreshed access token
_cached_token = {"access_token": None, "expires_at": 0}


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

    # Log ComfyUI output in background thread
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

        # Check if process crashed
        if comfyui_process and comfyui_process.poll() is not None:
            print(f"[handler] ComfyUI crashed with code {comfyui_process.returncode}")
            return False

        time.sleep(2)

    print(f"[handler] ComfyUI startup timeout ({COMFYUI_STARTUP_TIMEOUT}s)")
    return False


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


def queue_workflow(workflow):
    """Send workflow to ComfyUI's /prompt endpoint."""
    client_id = str(uuid.uuid4())

    payload = {
        "prompt": workflow,
        "client_id": client_id,
    }

    # Inject ComfyUI auth token for API nodes (GeminiImage2Node etc.)
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
        error_text = resp.text
        raise RuntimeError(f"Workflow queue failed ({resp.status_code}): {error_text}")


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


def collect_output_images(history_entry, only_nodes=None):
    """Collect output images from ComfyUI history entry as base64.

    Args:
        history_entry: ComfyUI history entry dict.
        only_nodes: Optional set of node IDs to collect. If None, collect all.
    """
    outputs = history_entry.get("outputs", {})
    images = []

    for node_id, node_output in outputs.items():
        if only_nodes and node_id not in only_nodes:
            print(f"[handler] Skipping node={node_id} (not in filter)")
            continue

        if "images" in node_output:
            for img_info in node_output["images"]:
                filename = img_info.get("filename")
                subfolder = img_info.get("subfolder", "")
                img_type = img_info.get("type", "output")

                # Fetch image from ComfyUI
                params = {
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": img_type,
                }
                resp = requests.get(
                    f"{COMFYUI_HOST}/view",
                    params=params,
                    timeout=30,
                )

                if resp.status_code == 200:
                    img_b64 = base64.b64encode(resp.content).decode("utf-8")
                    images.append({
                        "filename": filename,
                        "node_id": node_id,
                        "image": img_b64,
                    })
                    print(f"[handler] Collected: node={node_id}, file={filename}")
                else:
                    print(f"[handler] Failed to fetch {filename}: {resp.status_code}")

    return images


def handler(job):
    """RunPod serverless handler — main entry point."""
    print(f"[handler] === Job received ===")
    job_input = job.get("input", {})
    print(f"[handler] Input keys: {list(job_input.keys())}")

    # --- Validate input ---
    workflow = job_input.get("workflow")
    if not workflow:
        print("[handler] ERROR: Missing 'workflow' in input")
        return {"error": "Missing 'workflow' in input"}

    input_images = job_input.get("images", [])

    try:
        # --- Upload input images ---
        uploaded_names = {}
        for img in input_images:
            name = img.get("name", f"input_{uuid.uuid4().hex[:8]}.png")
            image_data = img.get("image", "")
            if image_data:
                uploaded_name = upload_image(name, image_data)
                uploaded_names[name] = uploaded_name

        # --- Patch workflow with uploaded image names ---
        if uploaded_names:
            for node_id, node in workflow.items():
                if node.get("class_type") == "LoadImage":
                    inputs = node.get("inputs", {})
                    original_name = inputs.get("image", "")
                    if original_name in uploaded_names:
                        # Exact name match
                        inputs["image"] = uploaded_names[original_name]
                    elif len(uploaded_names) == 1:
                        # Single image uploaded — apply to all LoadImage nodes
                        inputs["image"] = list(uploaded_names.values())[0]

        # --- Queue workflow ---
        prompt_id, client_id = queue_workflow(workflow)

        # --- Wait for completion ---
        execution_timeout = job_input.get("timeout", 600)
        history_entry = wait_for_completion(prompt_id, timeout=execution_timeout)

        # --- Collect outputs ---
        # Only collect needed nodes to keep response under RunPod size limit:
        #   68 = SaveImage (final composited result)
        #   41 = PreviewImage (face crop before enhancement)
        #   40 = PreviewImage (Gemini enhanced face)
        output_nodes = job_input.get("output_nodes")
        only_nodes = set(output_nodes) if output_nodes else {"68", "41", "40"}
        images = collect_output_images(history_entry, only_nodes=only_nodes)

        if not images:
            return {"error": "Workflow completed but no output images found"}

        return {
            "status": "success",
            "images": images,
        }

    except Exception as e:
        print(f"[handler] Error: {e}")
        return {"error": str(e)}


# --- Startup ---
if __name__ == "__main__":
    # Copy ComfyUI to workspace if needed (first run with volume)
    workspace_comfyui = "/workspace/ComfyUI"
    docker_comfyui = "/opt/ComfyUI"

    if not os.path.exists(workspace_comfyui) and os.path.exists(docker_comfyui):
        print("[handler] First run — copying ComfyUI to /workspace...")
        os.system(f"cp -r {docker_comfyui} {workspace_comfyui}")

    # Use workspace copy if available
    if os.path.exists(workspace_comfyui):
        os.environ["COMFYUI_PATH"] = workspace_comfyui
        COMFYUI_PATH = workspace_comfyui

    # Start ComfyUI
    start_comfyui()

    if not wait_for_comfyui():
        print("[handler] FATAL: ComfyUI failed to start")
        sys.exit(1)

    print("[handler] Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
