import os
from typing import List
import torch


def get_gpus_choices() -> List[int]:

    num_of_gpus = torch.cuda.device_count()
    
    return list(range(num_of_gpus))