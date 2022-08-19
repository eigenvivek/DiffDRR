from pathlib import Path

import torch
from torch.utils.data import Dataset


class ParamDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = Path(data_path)

    def __len__(self):
        return len([x for x in self.data_path.iterdir() if x.is_dir()])

    def __getitem__(self, idx):
        img_path = self.data_path / str(idx) / "image.pt"
        params_path = self.data_path / str(idx) / "params.pt"
        image = torch.load(img_path)
        params = torch.load(params_path)
        return image, params
