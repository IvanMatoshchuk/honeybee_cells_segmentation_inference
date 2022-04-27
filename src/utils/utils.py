import yaml
from yaml.loader import SafeLoader

import json
import numpy as np
import random

from PIL import ImageColor
import matplotlib.patches as mpatches

from typing import List, Tuple
import torch


def get_gpus_choices() -> List[int]:

    num_of_gpus = torch.cuda.device_count()

    return list(range(num_of_gpus))


def read_config(config_path: str) -> dict:

    stream = open(config_path, "r")
    config = yaml.load(stream, Loader=SafeLoader)

    return config


def read_label_classes(label_classes_path: str) -> dict:

    with open(label_classes_path, "r") as f:
        label_classes = json.load(f)
    return label_classes


def seed_everything(seed: int) -> None:

    """
    For reproducibility
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cmap_and_labels_for_plotting(label_classes_path: str) -> Tuple[dict, List[mpatches.Patch]]:

    # read label classes config (extracted from hasty.ai after labeling)
    label_classes = read_label_classes(label_classes_path)

    # prepare labels and colors
    cmap = {}
    labels = {}

    labels[0] = "background"
    for i, alpha in zip(label_classes, np.linspace(0.5, 1, len(label_classes))):
        color = ImageColor.getcolor(i["color"], "RGB")

        cmap[i["png_index"]] = [i / 255 for i in color] + [alpha]
        labels[i["png_index"]] = i["name"]

    cmap[0] = [1, 1, 1, 1]

    # create patches as legend
    patches = [mpatches.Patch(color=cmap[i], label=labels[i]) for i in cmap]

    return cmap, patches
