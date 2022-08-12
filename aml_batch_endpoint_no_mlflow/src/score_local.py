"""Code that helps us test our neural network before deploying to the cloud."""

import logging
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import numpy as np

from utils_score_nn import predict
from neural_network import NeuralNetwork

IMAGES_DIR = "aml_batch_endpoint_no_mlflow/test_data/images/"
MODEL_DIR = "aml_batch_endpoint_no_mlflow/model/weights.pth"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = MODEL_DIR
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    image_paths = [f for f in Path(IMAGES_DIR).iterdir() if Path.is_file(f)]
    image_paths.sort()
    images = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            x = np.array(image).reshape(1, -1) / 255.0
            images.append(x)

    dataloader = DataLoader(images)
    predicted_indices = predict(device, dataloader, model)
    predictions = [
        FashionMNIST.classes[predicted_index]
        for predicted_index in predicted_indices
    ]

    logging.info("Predictions: %s", predictions)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
