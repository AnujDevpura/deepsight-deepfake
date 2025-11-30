# 👁️ DeepSight: Deepfake Detection

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)

> A high-performance deep learning system that detects synthetic video content (deepfakes) with **98.1% accuracy** using EfficientNet-B3 and MTCNN face detection.

---

## Overview

DeepSight is a production-ready deepfake detection system that combines state-of-the-art computer vision models to identify manipulated video content. Built with PyTorch, it features both a user-friendly web interface and a REST API for seamless integration into existing workflows.

**Use Cases:**
- Media verification and fact-checking
- Social media content moderation
- Digital forensics and investigation
- Identity verification systems

---

## ✨ Key Features

- **🎯 High Accuracy** - 98.1% accuracy with F1-score of 0.99 on fake class detection
- **🔍 Dual-Stage Pipeline** - MTCNN for robust face extraction + EfficientNet for classification
- **⚡ Real-time Processing** - Optimized inference with mixed precision training (AMP)
- **🖥️ Interactive Dashboard** - Streamlit-based UI for easy video analysis
- **🔌 REST API** - FastAPI backend for programmatic access and integration
- **📊 Advanced Training** - Class-balanced weighted loss and learning rate scheduling
- **🎨 Multiple Models** - Support for EfficientNet-B0 and B3 architectures

---

## 📊 Performance

Evaluated on a test dataset of 31,143 samples:

| Model | Accuracy | Precision (Fake) | Recall (Fake) | F1-Score | Use Case |
|-------|----------|------------------|---------------|----------|----------|
| **EfficientNet-B3** | **98.10%** | **99.24%** | **98.32%** | **0.9878** | Production (Best) |
| EfficientNet-B0 | 74.00% | 68.00% | 90.00% | 0.7700 | Fast inference |

**Key Metrics:**
- False Positive Rate: < 1%

---

## 🏗️ System Architecture

```
Video Input → Frame Extraction → Face Detection (MTCNN) → Classification (EfficientNet) → Aggregation → Verdict
```

**Pipeline Components:**

1. **Preprocessing Module**
   - Extracts frames from video at configurable FPS
   - MTCNN detects and crops faces with alignment
   - Handles multiple faces per frame

2. **Inference Engine**
   - PyTorch-based EfficientNet classifier
   - Processes cropped face images
   - Returns probability scores per frame

3. **Aggregation Layer**
   - Averages predictions across all detected faces
   - Applies confidence thresholding
   - Outputs final REAL/FAKE verdict with confidence score

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)

### Install uv

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# With pip
pip install uv
```

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/anujdevpura/deepsight-deepfake.git
   cd deepsight-deepfake
   ```

2. **Install dependencies with uv**
   ```bash
   uv pip install -r requirements.txt
   ```

3. **Download model weights**
   
   Place your trained model weights in the `models/` directory:
   ```bash
   models/
   └── best_model_efficientnet_b3.pth
   ```
   
   Alternatively, train your own model (see [Training](#-training-your-own-model)).

---

## 🎮 Quick Start

DeepSight runs as a decoupled frontend and backend system. Launch both services:

### Option 1: Using uv run (Recommended)

**Terminal 1 - Start FastAPI Backend:**
```bash
uv run uvicorn app.main:app --reload --port 8000
```

**Terminal 2 - Start Streamlit Dashboard:**
```bash
uv run streamlit run app/dashboard.py
```

### Option 2: After Environment Activation

```bash
# Activate environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate    # Windows

# Terminal 1
uvicorn app.main:app --reload --port 8000

# Terminal 2
streamlit run app/dashboard.py
```

**Access Points:**
- Dashboard: http://localhost:8501
- API Documentation: http://localhost:8000/docs

### Using the Dashboard

1. Upload a video file (MP4, AVI, MOV)
2. Click "Analyze Video"
3. View results with confidence scores and frame-by-frame analysis
4. Download detailed report (optional)

---

## 🧠 Training Your Own Model

Train DeepSight on custom datasets (DFDC, FaceForensics++, or your own data).

### Step 1: Prepare Dataset

Organize videos in the following structure:

```
data/
├── real/   # Real human videos (.mp4)
└── fake/   # Deepfake videos (.mp4)
```

### Step 2: Extract Faces

Run preprocessing to extract and crop faces from videos:

```bash
cd scripts
uv run python preprocess_videos.py \
    --data_dir ../data \
    --output_dir ../data/frames_cropped_full \
    --max_frames 100
```

**Options:**
- `--max_frames`: Maximum frames to extract per video
- `--min_face_size`: Minimum face size for detection (default: 20)
- `--workers`: Number of parallel workers

### Step 3: Train Model

Launch training with automatic checkpointing and validation:

```bash
uv run python train.py \
    --model_name efficientnet_b3 \
    --epochs 10 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --use_amp
```

**Training Arguments:**
- `--model_name`: `efficientnet_b0` or `efficientnet_b3`
- `--use_amp`: Enable mixed precision training (faster on GPU)
- `--class_weights`: Apply weighted loss for imbalanced datasets
- `--scheduler`: Learning rate scheduler (`step`, `cosine`, `plateau`)

**Monitor Training:**
```bash
tensorboard --logdir runs/
```

### Step 4: Evaluate Model

Test model performance on held-out data:

```bash
uv run python scripts/evaluate.py \
    --model_path models/best_model_efficientnet_b3.pth \
    --test_dir data/test \
    --output_report reports/evaluation.txt
```

---

## 📂 Project Structure

```
deepsight-deepfake/
├── app/
│   ├── dashboard.py          # Streamlit web interface
│   ├── main.py              # FastAPI backend server
│   ├── inference.py         # Core inference engine
│   └── utils_video.py       # Video processing utilities
├── models/                  # Model weights (.pth files)
│   └── best_model_efficientnet_b3.pth
├── scripts/
│   ├── preprocess_videos.py # Face extraction pipeline
│   ├── train.py             # Training loop with AMP
│   └── evaluate.py          # Model evaluation script
│   └── predict_video.py     # Video prediction script
├── notebooks/               # Jupyter notebooks for experiments
├── reports/                 # Evaluation reports and metrics
├── data/                    # Dataset directory
│   └── fake                 # Fake videos
│   └── real                 # Real videos
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

---

## 🔧 Configuration

### Model Configuration

Edit `app/inference.py` to customize:

```python
# Model settings
MODEL_NAME = "efficientnet_b3"
CONFIDENCE_THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Preprocessing
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
```

### Video Processing

Edit `app/utils_video.py`:

```python
# Face detection
MIN_FACE_SIZE = 20
THRESHOLDS = [0.6, 0.7, 0.7]  # MTCNN thresholds

# Frame extraction
FRAMES_PER_VIDEO = 100
```

---

## Dependencies

**Datasets:**
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [DFDC](https://ai.facebook.com/datasets/dfdc/)
- [Celeb DF (v2)](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2)

**Frameworks:**
- [PyTorch](https://pytorch.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [MTCNN](https://github.com/ipazc/mtcnn)

**Research:**
- EfficientNet: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- MTCNN: [Zhang et al., 2016](https://arxiv.org/abs/1604.02878)

---

**⭐ If you find DeepSight useful, please consider giving it a star!**
