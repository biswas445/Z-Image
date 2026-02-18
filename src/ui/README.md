# Z-Image UI

Web interface for generating images from text using Z-Image.

---

## Quick Start

```bash
python -m ui.app
```

Open browser: **http://localhost:7860**

---

## Features

| Feature | Range | Default |
|---------|-------|---------|
| Height | 256-2048 | 1024 |
| Width | 256-2048 | 1024 |
| Steps | 1-50 | 8 |
| Guidance | 0-20 | 0.0 |
| Images | 1-4 | 1 |

- Prompt and negative prompt
- Seed control (-1 for random)
- CFG normalization option
- Auto-save to `outputs/ui/`

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ZIMAGE_UI_PORT` | 7860 | Server port |
| `ZIMAGE_UI_HOST` | 0.0.0.0 | Server host |
| `ZIMAGE_MODEL_PATH` | ckpts/Z-Image-Turbo | Model folder |
| `ZIMAGE_OUTPUT_DIR` | outputs/ui | Save folder |
| `ZIMAGE_DEFAULT_HEIGHT` | 1024 | Default height |
| `ZIMAGE_DEFAULT_WIDTH` | 1024 | Default width |
| `ZIMAGE_DEFAULT_STEPS` | 8 | Default steps |
| `ZIMAGE_ATTENTION` | _native_flash | Attention backend |
| `ZIMAGE_DTYPE` | bfloat16 | Model precision |
| `ZIMAGE_COMPILE` | false | Enable torch.compile |
| `ZIMAGE_AUTH_USERNAME` | - | Login username |
| `ZIMAGE_AUTH_PASSWORD` | - | Login password |

### Example

```bash
set ZIMAGE_UI_PORT=8080
set ZIMAGE_DEFAULT_HEIGHT=512
set ZIMAGE_DEFAULT_WIDTH=512
python -m ui.app
```

---

## Project Structure

```
src/ui/
├── __init__.py      # Module exports
├── config.py        # Settings and env vars
├── service.py       # Image generation logic
├── app.py           # Gradio web interface
└── README.md        # This file
```

---

## Architecture

```
┌─────────────────────────────────────┐
│  Interface: app.py (Gradio UI)      │
├─────────────────────────────────────┤
│  Orchestration: service.py          │
│  - GenerationRequest                │
│  - ImageGenerationService           │
├─────────────────────────────────────┤
│  Domain: config.py (UIConfig)       │
├─────────────────────────────────────┤
│  Infrastructure: zimage/, utils/    │
└─────────────────────────────────────┘
```

---

## API Usage

```python
from ui.config import UIConfig
from ui.service import GenerationRequest, ImageGenerationService

config = UIConfig()
service = ImageGenerationService(config)

request = GenerationRequest(
    prompt="A beautiful sunset",
    height=512,
    width=512,
    steps=8,
    seed=42,
)

result = service.generate(request)
print(f"Generated in {result.generation_time:.2f}s")
print(f"Saved to: {result.output_paths}")
```

---

## Tests

```bash
# Unit tests
python tests\test_ui.py

# Integration tests
python tests\test_integration.py
```

All 17 tests pass.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Port in use | Set `ZIMAGE_UI_PORT` to different value |
| Model not found | Download to `ckpts/Z-Image-Turbo` |
| Out of memory | Reduce height/width or steps |
| Slow generation | Use `_flash_3` backend |

---

## License

Same as Z-Image project.
