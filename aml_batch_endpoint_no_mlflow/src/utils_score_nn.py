"""Utilities that help with scoring neural networks."""

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


def predict(dataloader: DataLoader[np.ndarray[np.float64,
                                              np.dtype[np.float64]]],
            model: torch.nn.Module, device: str) -> list[torch.Tensor]:
    """
    Makes a prediction for the whole dataset once.
    """
    model.to(device)
    model.eval()

    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for x in dataloader:
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
