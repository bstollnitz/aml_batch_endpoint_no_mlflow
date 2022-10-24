"""Code that helps us test our neural network before deploying to the cloud."""

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST

from .dataset import FashionMNISTDatasetFromImages
from .neural_network import NeuralNetwork
from .utils_score_nn import predict

IMAGES_DIR = "aml_batch_endpoint_no_mlflow/test_data/images/"
MODEL_DIR = "aml_batch_endpoint_no_mlflow/model/weights.pth"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = MODEL_DIR
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    image_paths = [
        f.as_posix() for f in Path(IMAGES_DIR).iterdir() if Path.is_file(f)
    ]
    images_dataset = FashionMNISTDatasetFromImages(image_paths)

    dataloader = DataLoader(images_dataset)
    predicted_indices = predict(dataloader, model, device)
    predictions = [
        FashionMNIST.classes[predicted_index]
        for predicted_index in predicted_indices
    ]

    logging.info("Predictions: %s", predictions)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
