import os
import yaml

from typing import List
import torch


def get_gpus_choices() -> List[int]:

    num_of_gpus = torch.cuda.device_count()

    return list(range(num_of_gpus))


def read_config(config_path: str) -> dict:

    stream = open(config_path, "r")
    config = yaml.load(stream)

    return config
