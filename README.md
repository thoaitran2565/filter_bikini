# 👙 Automatic Bikini Image Classifier & Filter (PyTorch + CLIP)

An automated tool built with Python and PyTorch using OpenAI's **CLIP** model to automatically filter and organize bikini and swimwear photos from large datasets (e.g., 20,000+ images).

---

## ✨ Features

- **Semantic AI Matching:** Uses OpenAI's `clip-vit-base-patch32` for zero-shot image classification.
- **Negative Prompting:** Minimizes false positives by contrasting bikini labels against common non-bikini attire (e.g., gym clothes, dresses, casual wear).
- **GPU Acceleration:** Automatically leverages NVIDIA CUDA GPUs for fast processing.
- **Safe Copying:** Preserves original source files and safely copies matches to a destination folder.
- **Fault Tolerant:** Gracefully handles corrupted image files without interrupting the process.

---

## 🛠️ Prerequisites & Installation

### 1. Python Environment
Make sure you have Python **3.8** or higher installed.

### 2. Install Dependencies
Run the following command in your terminal to install the required libraries:

```bash
pip install torch torchvision pillow transformers tqdm
```

*(Note: For GPU acceleration, ensure PyTorch is installed with CUDA support).*

---

## 🚀 Usage

### 1. Configuration
Open `filter_bikini.py` and update the directory paths and parameters according to your setup:

```python
# Directory Configurations
SOURCE_DIR = r"D:\Code\tools\fillter_image\root"      # Path to your source image folder
OUTPUT_DIR = r"D:\Code\tools\fillter_image\result"    # Path where filtered images will be copied
THRESHOLD = 0.30                                   # Strictness score (recommended: 0.28 - 0.35)
```

### 2. Execution
Run the script using Python:

```bash
python filter_bikini.py
```

---

## ⚙️ How It Works

1. **Feature Extraction:** Loads images and processes them through the CLIP vision encoder.
2. **Text Contrast:** Evaluates the image against positive labels (`"a photo of a woman wearing a two-piece bikini"`) and negative labels (`"a woman wearing sports gym clothes"`).
3. **Filtering:** An image is selected only if the positive match score exceeds the defined `THRESHOLD` and outperforms all negative category scores.
4. **Export:** Matching images are safely duplicated to the `OUTPUT_DIR`.

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.