import os

import torch
import typer
from model import MyAwesomeModel

from data import load_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Starting evaluation")
    print(model_checkpoint)

    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    _, test_dataloader = load_data()

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    current_path = os.path.dirname(os.path.realpath(__file__))
    evaluate(current_path+"/../../models/model.pth")