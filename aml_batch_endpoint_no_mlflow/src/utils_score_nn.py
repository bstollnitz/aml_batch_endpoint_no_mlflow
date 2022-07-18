"""Utilities that help with scoring neural networks."""

import torch
from torch.utils.data import DataLoader


def predict(device: str, dataloader: DataLoader,
            model: torch.nn.Module) -> None:
    """
    Makes a prediction for the whole dataset once.
    """
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for (x,) in dataloader:
            tensor = x.float().to(device)
            predictions.extend(_predict_one_batch(model, tensor))
    return predictions


def _predict_one_batch(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Makes a prediction for input x.
    """
    with torch.no_grad():
        y_prime = model(x)
        probabilities = torch.nn.functional.softmax(y_prime, dim=1)
        predicted_indices = probabilities.argmax(1)
    return predicted_indices.cpu().numpy()
