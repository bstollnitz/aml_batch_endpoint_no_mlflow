"""Prediction."""

import argparse
import logging
import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST

from .dataset import FashionMNISTDatasetFromImages
from .neural_network import NeuralNetwork
from .utils_score_nn import predict


@dataclass
class State:
    model: torch.nn.Module
    device: str
    logger: logging.Logger


BATCH_SIZE = 64
state = None


def init() -> None:
    global state

    arg_parser = argparse.ArgumentParser(description="Argument parser.")
    arg_parser.add_argument("--logging_level", type=str, help="logging level")
    args, _ = arg_parser.parse_known_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(args.logging_level.upper())

    logger.info("Init started")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", default=""),
                              "model/weights.pth")
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    state = State(model, device, logger)

    logger.info("Init completed")


def run(mini_batch: list[str]) -> list[str]:
    if state is None:
        return []

    state.logger.info("run(%s started: %s", mini_batch, {__file__})

    images_dataset = FashionMNISTDatasetFromImages(mini_batch)
    dataloader = DataLoader(images_dataset)
    predicted_indices = predict(dataloader, state.model, state.device)
    predictions = [
        FashionMNIST.classes[predicted_index]
        for predicted_index in predicted_indices
    ]

    state.logger.info("Predictions: %s", predictions)
    state.logger.info("Run completed")

    return predictions
