import hashlib
import cv2
import numpy as np
from datasets.transforms import MEAN, STD, image_tensor


def corruption_transform(kind, seed=42):
    """In-memory test copies; never modifies disk or changes masks/geometry."""
    def transform(tensor, sample_id):
        image = np.clip((tensor.numpy().transpose(1, 2, 0) * STD + MEAN), 0, 1)
        if kind == 'brightness':
            image = image + 0.10
        elif kind == 'contrast':
            image = (image - 0.5) * 0.8 + 0.5
        elif kind == 'noise':
            digest = hashlib.sha256(f'{seed}:{sample_id}'.encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
            image = image + rng.normal(0, 0.03, image.shape)
        elif kind == 'blur':
            image = cv2.GaussianBlur(image, (3, 3), 0.8)
        else:
            raise ValueError(f'Unknown corruption: {kind}')
        return image_tensor(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8))
    return transform
