import os
from pathlib import Path

import cv2
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

project_path = Path(__file__).parent.parent.parent.parent


class CustomTestDataset(Dataset):
    def __init__(self, images_path: str = "data/images"):
        self.images_folder = os.path.join(project_path, images_path)

        self.images = [i for i in os.listdir(self.images_folder) if not i.startswith(".")]
        self.transforms = self.get_transforms()

    def get_transforms(self) -> A.core.composition.Compose:

        list_trans = [A.Normalize(mean=0, std=1), ToTensorV2()]
        list_trans = A.Compose(list_trans)
        return list_trans

    def __getitem__(self, index: int):

        img_path = os.path.join(self.images_folder, self.images[index])

        image = cv2.imread(img_path, 0)

        height = image.shape[0] // 32 * 32
        width = image.shape[1] // 32 * 32
        image = image[:height, :width]

        transformation = self.transforms(image=image)
        img_aug = transformation["image"]

        return img_aug, img_path

    def __len__(self) -> int:
        return len(self.images)
