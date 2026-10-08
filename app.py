import os
from pathlib import Path
from threading import Thread

import spaces
import torch
from fastapi.responses import HTMLResponse
from gradio import FileData, Server
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
    TextIteratorStreamer,
    set_seed,
)

ROOT = Path(__file__).parent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)

MODEL_NAME = "prithivMLmods/OneDecision-VisionGuard-9B-SFT" # / ** prithivMLmods/OneDecision-VisionGuard-27B-SFT / prithivMLmods/OneDecision-VisionGuard-4B-SFT ** /
PROMPT = "Classify this image as safe or unsafe."

set_seed(42)

print(f"Loading model: {MODEL_NAME} ...")
qwen_model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_NAME, attn_implementation="kernels-community/flash-attn2", torch_dtype=DTYPE, device_map=DEVICE,
).eval()
qwen_processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("Model loaded.")


def _file_path(f) -> str | None:
    """FileData arrives as a dict over the wire; tolerate the object form too."""
    if f is None:
        return None
    if isinstance(f, dict):
        return f.get("path")
    return getattr(f, "path", None)


app = Server(title="OneDecision VisionGuard")


@app.api(name="classify")
@spaces.GPU(size="xlarge")
def classify(image: FileData) -> str:
    """Streams the model's JSON classification of the uploaded image."""
    path = _file_path(image)
    if not path:
        raise ValueError("Please upload an image.")

    pil = Image.open(path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = qwen_processor(
        text=[text], images=[pil], return_tensors="pt", padding=True
    ).to(qwen_model.device)

    streamer = TextIteratorStreamer(
        qwen_processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=120,
    )
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=2048,
        use_cache=True,
        temperature=1.5,
        min_p=0.1,
    )
    thread = Thread(target=qwen_model.generate, kwargs=generation_kwargs)
    thread.start()

    full_text = ""
    for tok in streamer:
        full_text += tok
        yield full_text
    thread.join()


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((ROOT / "index.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    app.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )