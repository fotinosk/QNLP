import typing
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image


class Encoder(ABC):
    @abstractmethod
    def encode(self, image_path: Path) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def encode_batch(self, directory_path: Path) -> typing.Dict[str, np.array]:
        raise NotImplementedError

    def decode(self, image_path: Path) -> np.array:
        # Quantum approaches are not expected to be able to do this
        raise NotImplementedError

    @staticmethod
    def image_path_to_np_array(image_path: Path, grayscale: bool = False) -> np.array:
        img = Image.open(image_path)
        if grayscale:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        return np.array(img)
