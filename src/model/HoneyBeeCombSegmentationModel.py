import os
from pathlib import Path
from collections import OrderedDict

import torch
from torch import Tensor
import torch.nn as nn
import segmentation_models_pytorch as smp

project_path = Path(__file__).parent.parent.parent

mapping_dict = {"effnetb0": "efficientnet-b0", "mobilenetv3": "timm-mobilenetv3_small_100", "resnet18": "resnet18"}

PLACEHOLDER = 2


class HoneyBeeCombSegmentationModel:
    def __init__(self, model_name: str, device: str = "cpu"):

        self.model = self.get_model_by_name(model_name)

        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:

        return self.model(x)

    def get_model_by_name(self, model_name: str) -> PLACEHOLDER:

        path_to_state_dict = os.path.join(project_path, "models", model_name + ".pth")

        architecture, encoder = model_name.split("_")

        model = self._setup_model(architecture, encoder)
        state_dict = self._get_state_dict(path_to_state_dict)

        model.load_state_dict(state_dict)

        return model

    def _setup_model(architecture: str, encoder: str) -> PLACEHOLDER:

        if architecture == "unet":
            model = smp.Unet(encoder_name=encoder, classes=0, in_channels=1)
        elif architecture == "manet":
            model = smp.MAnet(encoder_name=encoder, classes=0, in_channels=1)
        elif architecture == "deeplabv3":
            model = smp.DeepLabV3(encoder_name=encoder, classes=0, in_channels=1)
        elif architecture == "deeplabv3p":
            model = smp.DeepLabV3Plus(encoder_name=encoder, classes=0, in_channels=1)
        else:
            raise Exception(f"selected model architecture: <{architecture}> is not defined")

        return model

    def _get_state_dict(path_to_state_dict: str) -> OrderedDict:

        pretrained_model = torch.load(path_to_state_dict, map_location=device)
        new_state_dict = OrderedDict()

        for k, v in pretrained_model["state_dict"].items():

            if k.split(".")[0] == "criterion":
                continue
            else:
                new_state_dict[k[6:]] = pretrained_model["state_dict"][k]

        return new_state_dict
