# Face Enhancer — RunPod Serverless Handler

ComfyUI-based пайплайн для AI-улучшения лиц. Работает на RunPod Serverless с GPU.

## Что делает

Оркестрирует трёхшаговый пайплайн:
1. **SAM3 Segmentation** — находит лицо, выдаёт маску и bounding box
2. **Python Square Crop** — вырезает квадратный кроп вокруг лица с padding и edge-reflect
3. **Gemini Image Generation** — улучшает кроп через Google Gemini

## Режимы работы

### detect
SAM3 сегментация → превью маски (красный оверлей).

**Вход:** image (base64), segment_pick (int)
**Выход:** PNG-превью с полупрозрачной красной маской

### enhance
Полный пайплайн: SAM3 → crop → Gemini → возврат всех артефактов.

**Вход:** image, segment_pick, model, resolution, prompt
**Выход:** улучшенный кроп (PNG/JPEG), маска (PNG), bbox, crop_info

### workflow (legacy)
Произвольный ComfyUI workflow JSON. Обратная совместимость.

## API контракт

```json
// Вход
{
  "input": {
    "mode": "detect" | "enhance" | "workflow",
    "image": "<base64>",
    "image_name": "photo.png",
    "segment_pick": 1,
    "model": "Nano Banana 2 (Gemini 3.1 Flash Image)",
    "resolution": "2K",
    "prompt": "..."
  }
}

// Выход (enhance)
{
  "status": "success",
  "mode": "enhance",
  "bbox": [x, y, w, h],
  "crop_info": {"x": int, "y": int, "size": int},
  "images": [
    {"node_id": "16", "image": "<base64>", "type": "enhanced"},
    {"node_id": "5", "image": "<base64>", "type": "mask"}
  ]
}
```

## Встроенные workflow

Workflow определены как Python dict в handler.py (строки 67-185). Не загружаются из файлов.
Файлы в `workflows/` — reference-копии для отладки в ComfyUI GUI.

| Workflow | Ноды | Результат |
|----------|------|-----------|
| DETECT_WORKFLOW | LoadImage → SAM3(blur=4) → ImageAndMaskPreview → PreviewImage | Превью маски |
| SAM3_BBOX_WORKFLOW | LoadImage → SAM3(blur=0) → MaskPreview + BboxFromMask | Маска + bbox |
| GEMINI_WORKFLOW | LoadImage → GeminiImage2Node → PreviewImage | Улучшенный кроп |

## Сжатие изображений

- Маски: всегда PNG (нужны чистые края)
- Превью: всегда PNG (JPEG ломает полупрозрачность)
- Улучшенное изображение: PNG если <10MB, JPEG (quality=95) если >10MB
- RunPod лимит ответа: ~20MB

## Переменные окружения

| Переменная | Default | Описание |
|-----------|---------|----------|
| COMFYUI_PATH | /workspace/ComfyUI | Путь к ComfyUI |
| COMFYUI_PORT | 8188 | Порт |
| COMFYUI_STARTUP_TIMEOUT | 120 | Таймаут запуска (сек) |
| COMFYUI_ARGS | "" | Доп. CLI аргументы |
| FIREBASE_API_KEY | "" | Firebase для Gemini нод |
| COMFY_REFRESH_TOKEN | "" | Firebase refresh token |

## Docker

### Автоматическая сборка (CI/CD)

Push в main → GitHub Actions → Docker build → DockerHub push.

```bash
git push origin main
# Ждать ~20-40 мин
# Затем на RunPod: убить воркер → создать нового
```

### Ручная сборка (на RunPod Pod)

```bash
bash build_and_push.sh <dockerhub_user>
```

### Что внутри Docker-образа

- PyTorch 2.10 + CUDA 12.6 + cuDNN 9
- ComfyUI (latest)
- 11 custom nodes: ComfyUI-RMBG, KJNodes, Easy-Use, rgthree, VideoHelperSuite, controlnet_aux, WanVideoWrapper, essentials, comfy_mtb, Manager, Custom-Scripts
- SAM3 модель (3.3GB, предзагружена)
- RunPod SDK

## Настройка RunPod Endpoint

| Параметр | Значение |
|----------|----------|
| Container Image | `art1aist/comfyui-face-enhancer-serverless:latest` |
| GPU | L4 24GB (или любой с 8GB+ VRAM) |
| Min Workers | 0 (Flex, scale to zero) |
| Max Workers | 1-5 |
| Idle Timeout | 30 сек |
| Execution Timeout | 600 сек |
| Container Disk | 20 GB |
