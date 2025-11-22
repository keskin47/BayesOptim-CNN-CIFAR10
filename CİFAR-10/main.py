import os
import json
import time
import torch
import matplotlib.pyplot as plt

from configs.config import Config
from models.mycnn import MyCNN
from trainer.trainer import Trainer
from bayes_opt import BayesOptimizer


# ================================================================
#  INFERENCE SCRIPT CREATOR
# ================================================================

def create_inference_script(path="inference.py"):
    content = """import torch
import torchvision.transforms as transforms
from PIL import Image
from models.mycnn import MyCNN

# Adjust params if needed
MODEL_PATH = "./checkpoints/cifar10_best_model.pth"

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    params = checkpoint.get("params", None)

    if params is None:
        raise ValueError("Params not found in checkpoint.")

    model = MyCNN(**params)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def preprocess_image(img_path):
    tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    ])
    img = Image.open(img_path).convert("RGB")
    return tf(img).unsqueeze(0)


def predict(img_path):
    model = load_model()
    x = preprocess_image(img_path)
    with torch.no_grad():
        outputs = model(x)
        pred = outputs.argmax(dim=1).item()
    return pred


if __name__ == "__main__":
    img_path = "test.jpg"
    print("Prediction:", predict(img_path))
"""

    with open(path, "w") as f:
        f.write(content)

    print(f"[Main] Auto-generated inference.py")


# ================================================================
#  TRAINING CURVE PLOT
# ================================================================

def plot_training_curves(log_path, output_path="results/training_curve.png"):
    if not os.path.exists(log_path):
        print("[Main] No log file found to plot.")
        return

    with open(log_path, "r") as f:
        logs = json.load(f)

    epochs = [e["epoch"] for e in logs]
    train_loss = [e["train_loss"] for e in logs]
    train_acc = [e["train_acc"] for e in logs]
    val_acc = [e["val_acc"] for e in logs]

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_acc, label="Train Acc")
    plt.plot(epochs, val_acc, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[Main] Training curve saved → {output_path}")




# ================================================================
#  MAIN PIPELINE
# ================================================================

def main():
    print("======================================")
    print("       CIFAR10 – Training Pipeline")
    print("======================================\n")

    # --------------------------------------------------
    # 1. CONFIG
    # --------------------------------------------------
    config = Config()
    print("[Main] Config loaded.")
    print(f"[Main] Device = {config.device}")

    # --------------------------------------------------
    # 2. BO SETUP
    # --------------------------------------------------
    bo = BayesOptimizer(
        config=config,
        trainer_cls=Trainer,
        model_cls=MyCNN
    )
    print("[Main] Bayesian Optimizer initialized.\n")

    # --------------------------------------------------
    # 3. RUN BO
    # --------------------------------------------------
    print("[Main] Running Bayesian Optimization...\n")
    start_time = time.time()

    best = bo.run(
        n_iterations=15,
        warmup=3
    )

    elapsed = time.time() - start_time
    print(f"\n[Main] BO completed in {elapsed/60:.2f} minutes.")
    print("Best Params:", best["params"])
    print("Best Validation Accuracy:", best["val_acc"])

    # --------------------------------------------------
    # 4. FINAL TRAINING
    # --------------------------------------------------
    print("\n======================================")
    print("       TRAINING FINAL MODEL...")
    print("======================================\n")

    final_params = best["params"]
    final_model = MyCNN(**final_params)

    trainer = Trainer(final_model, config)
    trainer.config.epochs = config.final_epochs

    final_acc = trainer.train()

    print(f"\n[Main] Final Model Accuracy: {final_acc:.4f}")

    # --------------------------------------------------
    # 5. PLOT TRAINING CURVES
    # --------------------------------------------------
    plot_training_curves(trainer.log_path)


    # --------------------------------------------------
    # 7. CREATE INFERENCE SCRIPT (if not exist)
    # --------------------------------------------------
    if not os.path.exists("inference.py"):
        create_inference_script()

    print("\n[Main] All tasks completed successfully!")


if __name__ == "__main__":
    main()
