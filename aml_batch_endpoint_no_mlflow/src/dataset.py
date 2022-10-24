"""Dataset created from Fashion MNIST images in the paths passed to the
constructor."""

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class FashionMNISTDatasetFromImages(Dataset[np.ndarray[np.float64,
                                                       np.dtype[np.float64]]]):
    """
    Dataset created from Fashion MNIST images in the paths passed to the
    constructor.
    """

    def __init__(self, images_paths: list[str]) -> None:
        self.image_paths = images_paths
        self.image_paths.sort()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self,
                    idx: int) -> np.ndarray[np.float64, np.dtype[np.float64]]:
        image_path = self.image_paths[idx]
        with Image.open(image_path) as image:
            x = np.array(image).reshape(1, -1) / 255.0
        return x
