import os
from pathlib import Path

import torch
from torch import Tensor
import segmentation_models_pytorch as smp

project_path = Path(__file__).parent.parent.parent

mapping_dict = {"effnetb0": "efficientnet-b0", "mobilenetv3": "timm-mobilenetv3_small_100", "resnet18": "resnet18"}


class HoneyBeeCombSegmentationModel:
    def __init__(self, model_name: str, device: str = "cpu"):

        self.model = self.get_model_by_name(model_name)

        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:

        return self.model(x)

    def get_model_by_name(self, model_name: str) -> torch.nn.Module:
        """
        initiates model from segmentation_models_pytorch package and loads pre-trained state-dict
        """

        path_to_state_dict = os.path.join(project_path, "models", model_name + ".pth")
        state_dict = torch.load(path_to_state_dict)

        architecture, encoder = model_name.split("_")
        encoder = mapping_dict[encoder]

        model = self._setup_model(architecture, encoder)
        model.load_state_dict(state_dict)

        return model

    def _setup_model(self, architecture: str, encoder: str) -> torch.nn.Module:

        if architecture == "unet":
            model = smp.Unet(encoder_name=encoder, classes=9, in_channels=1)
        elif architecture == "manet":
            model = smp.MAnet(encoder_name=encoder, classes=9, in_channels=1)
        elif architecture == "deeplabv3":
            model = smp.DeepLabV3(encoder_name=encoder, classes=9, in_channels=1)
        elif architecture == "deeplabv3p":
            model = smp.DeepLabV3Plus(encoder_name=encoder, classes=9, in_channels=1)
        else:
            raise Exception(f"selected model architecture: <{architecture}> is not defined!")

        return model
