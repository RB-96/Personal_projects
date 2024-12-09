import base64
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def read_image_from_path(image_path: Path, resize: Tuple[int, int] = None):
    image = cv2.imread(str(image_path))
    if resize:
        image = cv2.resize(image, resize)
    return image


def create_base64_image(image: np.ndarray):
    _, buffer = cv2.imencode(".jpg", image)
    image_base64 = base64.b64encode(buffer).decode("utf-8")
    return image_base64
