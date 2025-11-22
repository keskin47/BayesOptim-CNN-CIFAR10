import os
import json
import random
import numpy as np
import time
import torch


# ==========================================================
# SEED
# ==========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================================
# TIME TRACKER
# ==========================================================

class Timer:
    def __init__(self):
        self.start = time.time()

    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start


def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


# ==========================================================
# DIRECTORY HELPERS
# ==========================================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ==========================================================
# JSON HELPERS
# ==========================================================

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)
