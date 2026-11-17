import os
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
import albumentations as A
import cv2
import numpy as np
import torch

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def image_tensor(image):
    return torch.from_numpy(np.ascontiguousarray(((image.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)))


class TrainTransform:
    def __init__(self, size=384, foreground_probability=0.5, augment=True, reflect_mask_padding=False):
        self.size, self.fg_probability = size, foreground_probability
        self.reflect_mask_padding = reflect_mask_padding
        self.augment = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
            A.GaussNoise(std_range=(0.01, 0.03), p=0.15), A.GaussianBlur(blur_limit=(3, 3), p=0.15)]) if augment else None

    def __call__(self, image, mask):
        size = self.size
        ph, pw = max(0, size - mask.shape[0]), max(0, size - mask.shape[1])
        image = cv2.copyMakeBorder(image, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
        mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw,
                                 cv2.BORDER_REFLECT_101 if self.reflect_mask_padding else cv2.BORDER_CONSTANT, value=0)
        h, w = mask.shape
        if np.random.random() < self.fg_probability and mask.any():
            yy, xx = np.nonzero(mask)
            index = np.random.randint(len(yy))
            top = np.random.randint(max(0, yy[index] - size + 1), min(yy[index], h - size) + 1)
            left = np.random.randint(max(0, xx[index] - size + 1), min(xx[index], w - size) + 1)
        else:
            top, left = np.random.randint(h - size + 1), np.random.randint(w - size + 1)
        image, mask = image[top:top + size, left:left + size], mask[top:top + size, left:left + size]
        if self.augment:
            self.augment.set_random_seed(int(np.random.randint(2 ** 31)))
            result = self.augment(image=image, mask=mask)
            image, mask = result['image'], result['mask']
        return image_tensor(image), torch.from_numpy(np.ascontiguousarray(mask[None])).float()
