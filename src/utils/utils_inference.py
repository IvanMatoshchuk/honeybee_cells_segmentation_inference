import os
from pathlib import Path
import torch
import segmentation_models_pytorch as smp

project_path = Path(__file__).parent.parent.parent.parent

mapping_dict = {"effnetb0": "efficientnet-b0", "mobilenetv3": "timm-mobilenetv3_small_100", "resnet18": "resnet18"}

PLACEHOLDER = 2


def get_model_by_name(name: str) -> PLACEHOLDER:

    path_to_model = os.path.join(project_path, "models", name + ".pth")

    architecture, encoder = name.split("_")

    return


def setup_model(architecture: str, encoder: str) -> PLACEHOLDER:

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
