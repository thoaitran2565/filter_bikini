# 👙 Automatic Image Classifier & Filter (PyTorch + CLIP)

An automated high-performance tool built with Python, PyTorch, and OpenAI's **CLIP** model to automatically filter and organize photos from large datasets (e.g., 20,000+ images) based on semantic image content (Bikini/Swimwear and Nude/Explicit content).

---

## ✨ Features

- **Semantic AI Matching:** Powered by OpenAI's `clip-vit-base-patch32` for accurate zero-shot image classification.
- **Dual Filtering Modes:**
  - `filter_bikini.py`: Filters two-piece bikinis and swimwear.
  - `filter_nude_2.py`: Filters explicit nudity with high-threshold strictness.
- **Multithreaded Processing:** Utilizes Python's `ThreadPoolExecutor` for parallel image analysis to drastically boost throughput.
- **Incremental Scanning (History Tracker):** Keeps track of all processed image filenames in `processed_history.txt` / `processed_nude_history.txt`. New runs will skip previously scanned images without re-processing them.
- **Negative Prompting:** Minimizes false positives by contrasting target prompts against negative labels (e.g., gym clothes, dresses, casual attire, close-up faces).
- **GPU Acceleration:** Automatically leverages NVIDIA CUDA GPUs if available.
- **Non-Destructive:** Safe file operations that preserve original source files while copying matched outputs to dedicated folders.

---

## 🛠️ Prerequisites & Installation

### 1. Python Environment
Make sure you have Python **3.8** or higher installed.

### 2. Install Dependencies
Run the following command in your terminal to install the required libraries:

```bash
pip install torch torchvision pillow transformers tqdm
```

*(Note: For CUDA GPU acceleration, ensure PyTorch is installed with CUDA support).*

---

## 🚀 Usage

### 1. Filter Bikini / Swimwear

Configure directory paths in `filter_bikini.py`:

```python
SOURCE_DIR = r"D:\Code\tools\fillter_image\root"        # Path to source image folder
OUTPUT_DIR = r"D:\Code\tools\fillter_image\result"      # Destination for bikini photos
HISTORY_FILE = r"D:\Code\tools\fillter_image\processed_history.txt"
THRESHOLD = 0.30                                       # Strictness score
MAX_WORKERS = 4                                        # Parallel thread count
```

Run the script:
```bash
python filter_bikini.py
```

### 2. Filter Nude Content

Configure directory paths in `filter_nude_2.py`:

```python
SOURCE_DIR = r"D:\Ảnh\Ảnh đã tải"                      # Path to source image folder
OUTPUT_DIR = r"D:\Code\tools\fillter_image\result_nude" # Destination for nude photos
HISTORY_FILE = r"D:\Code\tools\fillter_image\processed_nude_history.txt"
THRESHOLD = 0.45                                       # Strictness score (recommended: 0.40 - 0.50)
MAX_WORKERS = 4                                        # Parallel thread count
```

Run the script:
```bash
python filter_nude_2.py
```

---

## ⚙️ How It Works

1. **History Check:** Reads existing history files to skip images that have already been evaluated in previous runs.
2. **Parallel Task Dispatch:** Distributes new images across `MAX_WORKERS` parallel threads.
3. **Feature Extraction:** Loads images into memory and processes them through the CLIP vision encoder.
4. **Contrastive Classification:** Compares image embeddings against positive and negative text prompts.
5. **Threshold Validation:** Copies an image to the destination directory if the positive score meets the `THRESHOLD` and exceeds negative match scores.
6. **Thread-Safe Logging:** Logs every evaluated image filename to prevent re-processing in future executions.

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.