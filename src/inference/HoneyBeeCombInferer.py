import os
from pathlib import Path
from typing import Optional, Union, Tuple

import cv2
import numpy as np
from tqdm import tqdm
import torch
from monai.inferers import SlidingWindowInferer
from torch import Tensor
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


import matplotlib.pyplot as plt


from src.dataset.CustomDataset import CustomDataset
from src.model.HoneyBeeCombSegmentationModel import HoneyBeeCombSegmentationModel
from src.utils.utils import read_config, get_cmap_and_labels_for_plotting, seed_everything

project_path = Path(__file__).parent.parent.parent
config_path_default = os.path.join(project_path, "config", "config.yaml")
label_classes_path_default = os.path.join(project_path, "data", "label_classes.json")
output_folder_for_masks_default = os.path.join(project_path, "data", "inferred_masks")
path_to_pretrained_models_default = os.path.join(project_path, "models")


class HoneyBeeCombInferer:
    def __init__(
        self,
        model_name: str,
        sw_inference: bool = True,
        device: str = "cpu",
        config_path: str = config_path_default,
        label_classes_path: str = label_classes_path_default,
        output_folder_for_masks: str = output_folder_for_masks_default,
        path_to_pretrained_models: str = path_to_pretrained_models_default,
    ):
        """
        class for performing semantic segmentation of honey bee comb

        Arguments:
            - model_name:

        """

        self.device = device
        self.sw_inferer = sw_inference
        self.output_folder_for_masks = output_folder_for_masks

        self.config = read_config(config_path)
        self.cmap, self.patches = get_cmap_and_labels_for_plotting(label_classes_path)

        self.model = HoneyBeeCombSegmentationModel(
            model_name=model_name, device=device, path_to_pretrained_models=path_to_pretrained_models
        )

        if sw_inference:
            self.sw_inferer = SlidingWindowInferer(**self.config["sliding_window_inferer"])

        seed_everything(self.config["random_seed"])

    def infer(self, image: Union[Tensor, np.array, str], return_logits: bool = False) -> Tensor:

        image = self._check_type_and_read_image_from_str(image)
        if isinstance(image, np.ndarray):
            image = self.preprocess_raw_image(image).to(self.device)
        if len(image.size()) < 4:
            image = image.unsqueeze(0)

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

        for image, image_path in tqdm(dataset):
            inferred_mask = self.infer(image.to(self.device))

            self._save_batch_inference(inferred_mask, image_path)

        return None

    def infer_without_bees(self, images_path: str, img_size=Tuple[int, int]) -> None:

        dataset = CustomDataset(images_path)
        dataloader = DataLoader(dataset=dataset, **self.config["dataloader"])

        output_means = []
        output = 0
        c = 0

        for image, image_path in tqdm(dataloader):
            inferred_logits = torch.softmax(
                self.infer(image.to(self.device), return_logits=True).detach(), dim=1
            ).cpu()

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

        return self._get_mask_no_bees(output_means, img_size)

    def _get_mask(self, inferred_logits: Tensor) -> Tensor:

        inferred_mask = torch.argmax(inferred_logits.squeeze(), dim=0).detach().cpu().numpy()

        return np.pad(inferred_mask, ((0, self.diff_height), (0, self.diff_width)))

    def _get_mask_no_bees(self, inferred_logits_means: Tensor, img_size: Tuple[int, int]) -> Tensor:

        inferred_logits_pred = torch.stack(inferred_logits_means)
        inferred_logits_pred = inferred_logits_pred.mean(dim=0)

        inferred_logits_pred = torch.softmax(inferred_logits_pred, dim=0).clone()
        inferred_mask = torch.argmax(inferred_logits_pred, dim=0).detach().cpu().numpy()

        diff_height = img_size[0] - inferred_mask.shape[0]
        diff_width = img_size[1] - inferred_mask.shape[1]

        return np.pad(inferred_mask, ((0, diff_height), (0, diff_width)))

    def preprocess_raw_image(self, image: np.array) -> Tensor:

        height = image.shape[0] // 32 * 32
        width = image.shape[1] // 32 * 32

        self.diff_height = image.shape[0] - height
        self.diff_width = image.shape[1] - width

        image = image[:height, :width]

        transformation = self.get_transforms()

        return transformation(image=image)["image"]

    def get_transforms(self) -> A.core.composition.Compose:

        list_trans = [A.Normalize(mean=0, std=1), ToTensorV2()]
        list_trans = A.Compose(list_trans)
        return list_trans

    def _adjust_class_weights(self, inferred_logits: Tensor) -> Tensor:

        inferred_logits[:, 1, ...] = 0
        inferred_logits[:, 0, ...] *= 0.35
        inferred_logits[:, 2, ...] *= 0.9

        return inferred_logits

    def _check_type_and_read_image_from_str(self, image: Union[str, np.ndarray, Tensor]) -> Union[np.ndarray, Tensor]:

        if isinstance(image, str):
            return cv2.imread(image, 0)
        else:
            return image

    def plot_prediction(
        self,
        pred: Tensor,
        input_image: Optional[Union[np.ndarray, str]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> None:

        label_processed = np.array([[self.cmap[int(i)] for i in j] for j in tqdm(pred)])

        if input_image is not None and mask is not None:
            fig, ax = plt.subplots(3, 1, figsize=(36, 28))

            input_image = self._check_type_and_read_image_from_str(input_image)
            ax[0].imshow(input_image, cmap="gray")
            ax[0].set_title("input image")

            ax[1].imshow(label_processed)
            ax[1].set_title("predicted")

            mask_processed = np.array([[self.cmap[i] for i in j] for j in tqdm(mask)])
            ax[2].imshow(mask_processed)
            ax[2].set_title("ground truth")

        elif input_image is not None or mask is not None:
            fig, ax = plt.subplots(1, 2, figsize=(36, 28))

            if input_image is not None:
                input_image = self._check_type_and_read_image_from_str(input_image)
                ax[0].imshow(input_image, cmap="gray")
                ax[0].set_title("input image")
            elif mask is not None:
                mask_processed = np.array([[self.cmap[i] for i in j] for j in tqdm(mask)])
                ax[0].imshow(input_image, cmap="gray")
                ax[0].set_title("input image")

            ax[1].imshow(label_processed)
            ax[1].set_title("predicted")

        else:
            fig, ax = plt.subplots(1, 1, figsize=(28, 20))

            ax.imshow(label_processed)
            ax.set_title("predicted")

        plt.legend(handles=self.patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.show()

        return None

    def _save_batch_inference(self, pred: Tensor, input_image_path: str) -> None:

        output_path = input_image_path.replace("images", "inferred_masks")

        label_processed = np.array([[self.cmap[int(i)] for i in j] for j in pred])
        fig, ax = plt.subplots(1, 1, figsize=(24, 20))

        ax.imshow(label_processed)
        ax.set_title("predicted")
        plt.legend(handles=self.patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.savefig(output_path)
        plt.close(fig)

        return None
