# config.py

import os
import torch
import numpy as np
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Global settings
random_seed = 6904

# Image & dataset config
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_CLASSES = 10
NUM_WORKERS = 0

# Training config
NUM_EPOCHS = 12
WARMUP_EPOCHS = np.round(0.25 * NUM_EPOCHS)
INITIAL_LR = 0.01
LABEL_SMOOTHING = 0.05
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

# PCA config
PCA_VARIANCE_PERCENT = 0.75

# Dataset URLs and paths
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
DATASET_DIR = Path("./imagenette2-320")
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

# Output paths
OUTPUT_DIR = Path("outputs")
TREE_VISUALIZATION_PATH = OUTPUT_DIR / "decision_tree.svg"
HEATMAP_DIR = OUTPUT_DIR / "feature_heatmap_grids"
HIGHLIGHTED_DIR = OUTPUT_DIR / "highlighted_images"
COMPOSITE_DIR = OUTPUT_DIR / "composite_images"
VLM_DESCRIPTOR_DIR = OUTPUT_DIR / "vlm_descriptors"
VLM_DESCRIPTOR_JSON_PATH = VLM_DESCRIPTOR_DIR / "descriptor_summary.json"
VLM_DESCRIPTOR_ANALYSIS_PATH = VLM_DESCRIPTOR_DIR / "descriptor_analysis.json"

# VLM descriptor generation
VLM_ENABLED = os.getenv("VINCE_VLM_ENABLED", "1") == "1"
VLM_PROVIDER = os.getenv("VINCE_VLM_PROVIDER", "openai")
VLM_MODEL = os.getenv("VINCE_VLM_MODEL", "gpt-4.1-mini")
VLM_COMPONENTS_TO_DESCRIBE = int(os.getenv("VINCE_VLM_COMPONENTS_TO_DESCRIBE", "20"))
VLM_TOP_SAMPLES_PER_COMPONENT = int(os.getenv("VINCE_VLM_TOP_SAMPLES_PER_COMPONENT", "4"))
VLM_DESCRIPTOR_CONFIDENCE_THRESHOLD = float(os.getenv("VINCE_VLM_DESCRIPTOR_CONFIDENCE_THRESHOLD", "0.55"))
VLM_REQUEST_TIMEOUT_SEC = int(os.getenv("VINCE_VLM_REQUEST_TIMEOUT_SEC", "90"))

# Class names for Imagenette
IMAGENETTE_CLASSES = [
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute"
]

# Ensure output directories exist
for dir_path in [OUTPUT_DIR, HEATMAP_DIR, HIGHLIGHTED_DIR, COMPOSITE_DIR, VLM_DESCRIPTOR_DIR]:
    os.makedirs(dir_path, exist_ok=True)
