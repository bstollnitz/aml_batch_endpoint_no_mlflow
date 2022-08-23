"""Prediction."""

import argparse
import logging
import os
import numpy as np

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST

from neural_network import NeuralNetwork
from utils_score_nn import predict

logger = None
model = None
device = None
BATCH_SIZE = 64


def init():
    global logger
    global model
    global device

    arg_parser = argparse.ArgumentParser(description="Argument parser.")
    arg_parser.add_argument("--logging_level", type=str, help="logging level")
    args, _ = arg_parser.parse_known_args()
    logger = logging.getLogger(__name__)
    logger.setLevel(args.logging_level.upper())

    logger.info("Init started")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"),
                              "model/weights.pth")
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    logger.info("Init completed")


def run(mini_batch):
    logger.info("run(%s started: %s", mini_batch, {__file__})

    images = []
    for image_path in mini_batch:
        with Image.open(image_path) as image:
            x = np.array(image).reshape(1, -1) / 255.0
            images.append(x)

    dataloader = DataLoader(images)
    predicted_indices = predict(device, dataloader, model)
    predictions = [
        FashionMNIST.classes[predicted_index]
        for predicted_index in predicted_indices
    ]

    logger.info("Predictions: %s", predictions)

    logger.info("Run completed")
    return predictions
