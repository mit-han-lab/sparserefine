import os
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .utils import Normalize, Pad, PhotoMetricDistortion, RandomCrop, RandomFlip, Resize

__all__ = ["Cityscapes"]


class Cityscapes(Dataset):
    def __init__(self, root: str, prediction_dir: str, split: str) -> None:
        super().__init__()
        self.root = root
        self.prediction_dir = prediction_dir
        self.split = split

        self.data = []
        img_folder = os.path.join(root, "leftImg8bit", split)
        for _, _, files in os.walk(img_folder):
            for fname in files:
                if fname.startswith("._"):
                    continue
                if fname.endswith(".png"):
                    self.data.append(fname.split("_", 1)[0] + "/" + fname.rsplit("_", 1)[0])

        if split == "train":
            self.transforms = [
                Resize(img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
                RandomCrop(crop_size=(512, 1024), cat_max_ratio=0.75),
                RandomFlip(prob=0.5),
                PhotoMetricDistortion(),
                Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                Pad(size=(512, 1024), pad_val=0, seg_pad_val=255),
            ]
        else:
            self.transforms = [
                Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            ]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_path = os.path.join(self.root, "leftImg8bit", self.split, self.data[index] + "_leftImg8bit.png")
        image = np.array(Image.open(image_path))

        label_path = os.path.join(self.root, "gtFine", self.split, self.data[index] + "_gtFine_labelTrainIds.png")
        label = np.array(Image.open(label_path))

        logits_path = os.path.join(self.prediction_dir, self.split, self.data[index] + ".npy")
        
        if self.prediction_dir.split('/')[-2].startswith('mask2former'):
            probs = np.load(logits_path).astype(np.float32)
            probs = probs.transpose((1, 2, 0))
            probs = cv2.resize(probs, (2048, 1024), interpolation=cv2.INTER_NEAREST)
            probs = probs.clip(1e-10)

            logits = np.log(probs)

        else:
            logits = np.load(logits_path).astype(np.float32)
            logits = logits.transpose((1, 2, 0))
            logits = cv2.resize(logits, (2048, 1024), interpolation=cv2.INTER_NEAREST)

        infos = {}
        infos["img"] = image
        infos["label"] = label
        infos["logits"] = logits
        infos["seg_fields"] = ["label", "logits"]

        for transform in self.transforms:
            infos = transform(infos)

        image = infos["img"]
        label = infos["label"]
        logits = infos["logits"]

        return {"image": image, "logits": logits, "label": label}

    def __len__(self) -> int:
        return len(self.data)