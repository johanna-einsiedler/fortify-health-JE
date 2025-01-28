from fortify.data import load_data
from fortify.model import MyAwesomeModel
import torch
from torch import nn
from torchvision import datasets, transforms
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import typer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")



def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    """Train a model on MNIST."""
    print("Training started")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    model = MyAwesomeModel().to(DEVICE)

    train_dataloader,_ = load_data()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    torch.save(model.state_dict(), "models/model.pth")

if __name__ == "__main__":
    train()