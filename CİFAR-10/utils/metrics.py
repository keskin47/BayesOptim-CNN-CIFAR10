import torch
import torch.nn.functional as F


def accuracy(output, target):
    preds = output.argmax(dim=1)
    correct = (preds == target).sum().item()
    return correct / target.size(0)


def accuracy_topk(output, target, topk=(1, 5)):
    maxk = max(topk)
    _, pred = output.topk(maxk, dim=1)

    out = {}
    for k in topk:
        correct_k = pred[:, :k].eq(target.view(-1, 1)).sum().item()
        out[k] = correct_k / target.size(0)

    return out


class AverageMeter:
    """
    AverageMeter for tracking loss & accuracy.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count != 0 else 0


def confusion_matrix(preds, labels, num_classes=10):
    cm = torch.zeros(num_classes, num_classes)
    for p, t in zip(preds, labels):
        cm[t, p] += 1
    return cm
