import os
from statistics import mode
from typing import List, Optional, Union

import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from torch import Tensor
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from src.dataset.dataset import CustomDataset
from src.model.segmentation_model import HoneyBeeCombSegmentationModel
from src.utils.utils import read_config

project_path = Path(__file__).parent.parent.parent
config_path_default = os.path.join(project_path, "config", "config.yaml")


class HoneyBeeCombInferer:
    def __init__(
        self,
        path_to_images: str,
        model_name: str,
        config_path: str = config_path_default,
        sw_inference: bool = True,
        device: str = "cpu",
    ):

        self.path_to_images = path_to_images
        self.device = device
        self.sw_inferer = sw_inference

        self.config = read_config(config_path)

        self.model = HoneyBeeCombSegmentationModel(model_name=model_name, device=device)

        if sw_inference:
            self.sw_inferer = SlidingWindowInferer(**self.config["sliding_window_inferer"])

    def infer(self, image: Union[Tensor, np.array, str], return_logits: bool = False) -> Tensor:

        if isinstance(image, str):
            image = cv2.imread(image, 0)

        image = self.preprocess_raw_image(image).to(self.device)

        if self.sw_inferer:
            inferred_logits = self.sw_inferer(image, self.model)
        else:
            inferred_logits = self.model(image)

        if return_logits:
            return inferred_logits
        else:
            return self._get_mask(inferred_logits)

    def infer_batch(self, images_path: str) -> None:

        dataset = CustomDataset(images_path)
        dataloader = DataLoader(dataset=dataset, **self.config["dataloader"])

        for image, image_path in tqdm(dataloader):
            inferred_mask = self.infer(image)

            visualizer.save_pred(inferred_mask)

    def infer_without_bees(self, images_path: str) -> None:

        dataset = CustomDataset(images_path)
        dataloader = DataLoader(dataset=dataset, **self.config["dataloader"])

        output_means = []
        output = 0
        c = 0

        for image, image_path in tqdm(dataloader):
            inferred_logits = self.infer(image, return_logits=True)

            inferred_logits = self._adjust_class_weights(inferred_logits)

            if type(output) is int:
                output = inferred_logits.clone()
            else:
                output = torch.cat([inferred_logits, output])
                if c % 5 == 0:
                    output_means.append(output.mean(dim=0))
                    del output
                    output = 0

            c += 1

        output_means.append(output.mean(dim=0))

    def _get_mask(self, inferred_logits: Tensor) -> Tensor:

        return torch.argmax(inferred_logits.squeeze(), dim=0)

    def preprocess_raw_image(self, image: Union[Tensor, np.array]) -> Tensor:

        height = image.shape[0] // 32 * 32
        width = image.shape[1] // 32 * 32
        image = image[:height, :width]

        transformation = self.get_transforms()

        return transformation(image=image)["image"]

    def get_transforms(self) -> A.core.composition.Compose:

        list_trans = [A.Normalize(mean=0, std=1), ToTensorV2()]
        list_trans = A.Compose(list_trans)
        return list_trans

    def _adjust_class_weights(inferred_logits: Tensor) -> Tensor:

        inferred_logits[:, 1, ...] = 0
        inferred_logits[:, 0, ...] *= 0.35
        inferred_logits[:, 2, ...] *= 0.9

        return inferred_logits

    def _get_mask_no_bees(inferred_logits_means: Tensor) -> Tensor:

        inferred_logits_pred = torch.stack(inferred_logits_means)
        inferred_logits_pred = inferred_logits_pred.mean(dim=0)

        inferred_logits_pred = torch.softmax(inferred_logits_pred, dim=0).clone()

        return torch.argmax(inferred_logits_pred, dim=0)
