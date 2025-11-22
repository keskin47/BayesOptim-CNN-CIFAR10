import os
import json
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from utils.metrics import accuracy
from utils.data_utils import get_dataloaders, get_cifar10_datasets


class Trainer:
    def __init__(self, model, config, resume=False):
        self.config = config
        self.device = config.device
        self.model = model.to(self.device)

        # -----------------------------------------------------
        # PATHS
        # -----------------------------------------------------
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.logs_dir, exist_ok=True)

        self.last_ckpt_path = os.path.join(config.checkpoint_dir, "checkpoint_last.pth")
        self.best_ckpt_path = os.path.join(config.checkpoint_dir, "cifar10_best_model.pth")
        self.log_path = os.path.join(config.logs_dir, "train_log.json")

        # -----------------------------------------------------
        # GPU LOG
        # -----------------------------------------------------
        print(f"[Trainer] Device: {self.device}")
        if self.device == "cuda":
            print(f"[Trainer] GPU: {torch.cuda.get_device_name(0)}")

        # -----------------------------------------------------
        # Dataset
        # -----------------------------------------------------
        train_dataset, test_dataset = get_cifar10_datasets(
            root="./data",
            mean=config.mean,
            std=config.std,
            image_size=config.image_size,
            download=False
        )

        self.train_loader, self.test_loader = get_dataloaders(
            train_dataset,
            test_dataset,
            batch_size=model.batch_size,
            num_workers=config.num_workers
        )

        # -----------------------------------------------------
        # Optimizer
        # -----------------------------------------------------
        if model.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=model.learning_rate,
                weight_decay=model.weight_decay
            )
        elif model.optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=model.learning_rate,
                momentum=0.9,
                weight_decay=model.weight_decay
            )
        elif model.optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=model.learning_rate,
                weight_decay=model.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {model.optimizer_type}")

        print(f"[Trainer] Using optimizer = {model.optimizer_type}")

        # -----------------------------------------------------
        # LR Scheduler
        # -----------------------------------------------------
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            patience=2,
            factor=0.5,
            min_lr=1e-6
        )

        # -----------------------------------------------------
        # AMP (Mixed Precision)
        # -----------------------------------------------------
        self.scaler = GradScaler(enabled=(self.device == "cuda"))

        # -----------------------------------------------------
        # Early Stopping
        # -----------------------------------------------------
        self.patience = 4
        self.best_val_acc = 0.0
        self.wait = 0

        # -----------------------------------------------------
        # Resume
        # -----------------------------------------------------
        self.start_epoch = 0
        if resume and os.path.exists(self.last_ckpt_path):
            self._load_checkpoint()
            print("[Trainer] Resume → training will continue.")
        else:
            print("[Trainer] Starting fresh training.")

    # ============================================================== #
    #                    CHECKPOINT SYSTEM
    # ============================================================== #

    def _save_checkpoint(self, epoch, val_acc, is_best=False):
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
        }

        torch.save(ckpt, self.last_ckpt_path)

        if is_best:
            torch.save(ckpt, self.best_ckpt_path)

    def _load_checkpoint(self):
        ckpt = torch.load(self.last_ckpt_path, map_location=self.device)
        self.start_epoch = ckpt["epoch"] + 1
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.best_val_acc = ckpt["best_val_acc"]

    # ============================================================== #
    #                        TRAINING
    # ============================================================== #

    def train_one_epoch(self):
        self.model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # AMP
            with autocast(enabled=(self.device == "cuda")):
                outputs = self.model(images)
                loss = F.cross_entropy(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

            if batch_idx % self.config.print_every == 0:
                print(f"[Train] Batch {batch_idx}/{len(self.train_loader)} - Loss: {loss.item():.4f}")

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc

    def validate(self):
        self.model.eval()
        total_correct, total_samples = 0, 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                total_correct += (outputs.argmax(1) == labels).sum().item()
                total_samples += labels.size(0)

        return total_correct / total_samples

    # ============================================================== #
    #                     TRAIN ENTRY POINT
    # ============================================================== #

    def train(self):

        # ✔ FIX: config.epochs yoksa → BO epoch sayısını kullan
        if not hasattr(self.config, "epochs"):
            self.config.epochs = self.config.bo_epochs

        print(f"[Trainer] Training for {self.config.epochs} epochs")

        for epoch in range(self.start_epoch, self.config.epochs):

            print(f"\n========== EPOCH {epoch+1} ==========")

            train_loss, train_acc = self.train_one_epoch()
            val_acc = self.validate()

            print(f"[Epoch {epoch+1}] TrainAcc={train_acc:.4f}  ValAcc={val_acc:.4f}")

            # Scheduler güncelle
            self.scheduler.step(val_acc)

            # Log kaydı
            self._log_epoch(epoch, train_loss, train_acc, val_acc)

            # En iyi model?
            if val_acc > self.best_val_acc:
                print(f"[Trainer] New best model! ({val_acc:.4f})")
                self.best_val_acc = val_acc
                self.wait = 0
                self._save_checkpoint(epoch, val_acc, is_best=True)
            else:
                self.wait += 1

            # Last checkpoint
            self._save_checkpoint(epoch, val_acc, is_best=False)

            # Early stopping
            if self.wait >= self.patience:
                print(f"[EarlyStopping] No improvement. Best val_acc: {self.best_val_acc:.4f}")
                break

            torch.cuda.empty_cache()

        return self.best_val_acc

    # ============================================================== #
    #                       LOGGING
    # ============================================================== #

    def _log_epoch(self, epoch, loss, acc, val_acc):
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": loss,
            "train_acc": acc,
            "val_acc": val_acc
        }

        if not os.path.exists(self.log_path):
            logs = []
        else:
            with open(self.log_path, "r") as f:
                try:
                    logs = json.load(f)
                except:
                    logs = []

        logs.append(log_entry)

        with open(self.log_path, "w") as f:
            json.dump(logs, f, indent=4)
