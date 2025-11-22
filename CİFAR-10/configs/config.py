import os
import torch
from torch.utils.data import DataLoader
from utils.misc import set_seed


class Config:
    """
    Projedeki tüm ayarların merkezi kontrolü.
    Trainer + BO + Main pipeline ile tamamen uyumlu.
    """

    def __init__(self,
                 seed=42,
                 auto_compute_stats=False,
                 dataset=None):

        # ==========================================================
        # RANDOMNESS (deterministic behavior)
        # ==========================================================
        self.seed = seed
        set_seed(self.seed)

        # ==========================================================
        # DEVICE
        # ==========================================================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Config] Using device: {self.device}")

        # ==========================================================
        # PATHS (Trainer & BO tarafından kullanılıyor)
        # ==========================================================
        self.checkpoint_dir = "./checkpoints"
        self.logs_dir = "./logs"
        self.results_dir = "./results"

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # ==========================================================
        # DATASET SETTINGS
        # ==========================================================
        self.image_size = 32
        self.num_workers = 4
        self.print_every = 50

        # CIFAR10 default mean/std
        if dataset is not None and auto_compute_stats:
            print("[Config] Computing mean/std from dataset...")
            self.mean, self.std = self.compute_mean_std(dataset)
        else:
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std =  (0.2470, 0.2435, 0.2616)

        # ==========================================================
        # TRAINING SETTINGS
        # ==========================================================
        # BO hızlı çalışsın diye kısa epoch
        self.bo_epochs = 10

        # Final model epoch (sen 30 istedin)
        self.final_epochs = 30

        # ==========================================================
        # BAYESIAN OPTIMIZATION SEARCH SPACE
        # ==========================================================
        self.search_space = {
            # float range
            "learning_rate": (1e-5, 1e-2),

            # int range
            "batch_size": (32, 256),

            # CNN filters
            "filters1": (16, 128),
            "filters2": (16, 256),

            # float range
            "dropout": (0.0, 0.4),
            "weight_decay": (1e-6, 1e-3),

            # categorical
            "optimizer": ["adam", "sgd", "adamw"]
        }

    # ==============================================================
    # MEAN / STD COMPUTATION (optional)
    # ==============================================================
    def compute_mean_std(self, dataset):
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

        mean = 0.
        std = 0.
        total = 0

        for images, _ in loader:
            images = images.view(images.size(0), images.size(1), -1)
            batch_samples = images.size(0)

            mean += images.mean(dim=[0, 2]) * batch_samples
            std += images.std(dim=[0, 2]) * batch_samples
            total += batch_samples

        mean /= total
        std /= total

        print("[Config] mean:", mean.tolist())
        print("[Config] std: ", std.tolist())

        return mean.tolist(), std.tolist()



