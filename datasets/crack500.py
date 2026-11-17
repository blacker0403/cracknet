from pathlib import Path
import torch
from torch.utils.data import Dataset

from datasets.manifest import read_image, read_mask
from datasets.transforms import image_tensor


class Crack500(Dataset):
    def __init__(self, root, rows, label_map=None, transform=None):
        self.root, self.rows, self.label_map, self.transform = Path(root), rows, label_map, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image, mask = read_image(self.root / row['image']), read_mask(self.root / row['mask'], self.label_map)
        if image.shape[:2] != mask.shape:
            raise ValueError(f'Image/mask shape mismatch: {row["id"]}')
        x, y = self.transform(image, mask) if self.transform else (image_tensor(image), torch.from_numpy(mask[None]).float())
        sample = {'image': x, 'mask': y, 'id': row['id']}
        if 'domain' in row:
            sample['domain'] = row['domain']
        return sample
