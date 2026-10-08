# **OneDecision-VisionGuard-Demo**

OneDecision-VisionGuard-Demo is an interactive, visual content-moderation and guardrail workflow powered by the `prithivMLmods/OneDecision-VisionGuard-9B-SFT` vision-language classification model (with architectural compatibility extending across the `4B-SFT` and `27B-SFT` variants). Designed as an end-to-end trust and safety screening system, the platform processes visual media to output binary safety classifications alongside granular natural-language reasoning.

The workspace couples a FastAPI-backed Gradio Server engine with a canvas-rendered HTML5 single-page application (SPA). The interface presents a visual pipeline diagram depicting live image ingest, tokenization, model inspection, real-time token streaming, verdict routing (*Passed / safe = 1* versus *Dumped / nsfw = 1*), and contextual session history logging.

### **Key Features**

* **Vision-Language Safety Screening:** Evaluates source images with structured instruction prompting (`Classify this image as safe or nsfw.`) to determine compliance and generate chain-of-thought moderation rationale.
* **Qwen-VL Backbone Integration:** Leverages `Qwen3_5ForConditionalGeneration` alongside native attention acceleration via `kernels-community/flash-attn2` for low-latency visual tensor encoding.
* **Canvas Workflow Architecture:** A self-contained HTML5 `<canvas>` rendering pipeline visualizing the processing flow from ingestion to classification with dynamic token transitions and classification states.
* **Real-Time Token Streaming:** Employs `TextIteratorStreamer` within a background execution thread to continuously pipe model output tokens and JSON classification structures to the frontend.
* **Automated Content Redaction:** Automatically generates downscaled pixelated/blurred fallbacks when NSFW or sensitive media is detected, preventing raw exposure in production environments.

### **Repository Structure**

```text
├── app.py
├── index.html
├── LICENSE.txt
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock
```

### **Installation and Requirements**

To set up OneDecision-VisionGuard-Demo locally, configure your system according to the specifications below. A modern CUDA-enabled GPU with bfloat16 support is required.

* **Python Version:** Minimum Python **3.10.13** is required; Python **3.14** is recommended.
* **PyTorch Version:** `torch==2.13.0` or above is required for model execution and kernel compatibility.
* **CUDA Version:** **CUDA 13.0** is recommended, matching the execution environment on the live Hugging Face Space.

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust. It ensures rapid virtual environment setup and exact dependency synchronization based on the `uv.lock` file.

**Step 1 — Install `uv`**

* **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

**Step 2 — Clone the repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/OneDecision-VisionGuard-Demo.git
cd OneDecision-VisionGuard-Demo
```

**Step 3 — Initialize the project and install dependencies**

```bash
uv sync
```

**Step 4 — Run the application**

```bash
uv run app.py
```

#### **Standard PIP Implementation**

**1. Update Package Manager**
Upgrade your local package manager:

```bash
pip install "pip>=26.2.1"
```

**2. Install Core Dependencies**
Install the primary deep learning stack, transformer libraries, and core utilities listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

#### **Core Requirements List (`requirements.txt`)**

```text
torch==2.13.0
torchvision==0.28.0
transformers==5.18.0
accelerate==1.14.0
peft==0.19.1
gradio==6.29.1
av==17.1.0
spaces>=0.51.1
huggingface-hub>=1.24.0
pillow>=12.3.0
kernels>=0.17.2
```

### **Usage**

Once initialized, navigate to the local address output in your terminal (typically `http://127.0.0.1:7860/`).

1. **Load Image Media:** Click the **Upload image** area on the left to select a local file, drop an image onto the canvas, or click any of the safe reference images in the **Examples** tray.
2. **Execute Moderation Check:** Click the **Run** button to launch inference. The canvas will animate the image traveling through patchification, encoding, projection, and decoding phases.
3. **Inspect Output & Verdicts:**
* **Passed (`safe = 1`):** Safe content is approved and routed to the downstream release tray.
* **Dumped (`nsfw = 1`):** Flagged or unsafe content is blocked, redacted, and diverted to the quarantined discard tray.
* **Reasoning:** Review the raw JSON payload and the natural-language classification rationale displayed in the model output terminal.

### **License and Source**

* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/OneDecision-VisionGuard-Demo/blob/main/LICENSE.txt)
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/OneDecision-VisionGuard-Demo.git](https://github.com/PRITHIVSAKTHIUR/OneDecision-VisionGuard-Demo.git)
* **Hugging Face Live Space:** [https://huggingface.co/spaces/prithivMLmods/OneDecision-VisionGuard-Demo](https://huggingface.co/spaces/prithivMLmods/OneDecision-VisionGuard-Demo)
