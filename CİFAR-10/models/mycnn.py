import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    """
    Xavier initialization for Conv + Linear layers.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class MyCNN(nn.Module):
    def __init__(
        self,
        learning_rate,
        batch_size,
        optimizer,
        dropout,
        weight_decay,
        filters1,
        filters2
    ):
        super().__init__()

        # Hyperparams (BO’dan geliyor)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer_type = optimizer
        self.dropout_rate = dropout
        self.weight_decay = weight_decay

        self.filters1 = filters1
        self.filters2 = filters2

        # ============================================================
        #  BLOCK 1
        # ============================================================
        self.block1 = nn.Sequential(
            nn.Conv2d(3, filters1, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters1),
            nn.ReLU(inplace=True),

            nn.Conv2d(filters1, filters1, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2)   # 32 → 16
        )

        # ============================================================
        #  BLOCK 2
        # ============================================================
        self.block2 = nn.Sequential(
            nn.Conv2d(filters1, filters2, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters2),
            nn.ReLU(inplace=True),

            nn.Conv2d(filters2, filters2, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters2),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  # 16 → 8
            nn.Dropout(p=dropout)  # Conv dropout (daha iyi gen.)
        )

        # ============================================================
        #  CLASSIFIER
        # ============================================================
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters2 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 10)
        )

        # Weight initialization
        self.apply(init_weights)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x

    def summary(self):
        print("===== MyCNN Model Summary =====")
        print(f"filters1       : {self.filters1}")
        print(f"filters2       : {self.filters2}")
        print(f"dropout        : {self.dropout_rate}")
        print(f"optimizer      : {self.optimizer_type}")
        print(f"learning_rate  : {self.learning_rate}")
        print(f"batch_size     : {self.batch_size}")
        print(f"weight_decay   : {self.weight_decay}")
        print("================================")
