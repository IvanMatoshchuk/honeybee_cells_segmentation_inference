# Segmentation of Honey Bee Comb
Inference pipeline for performing segmentation of honey bee comb and counting of individual cells.\
Pipeline can be used for inferring all images in the specified folder (execution via command line) or used as a package in python scripts.

Pre-trained models can be downloaded from: https://www.dropbox.com/sh/2b8vcr0mwdsqwpb/AACLDpQ1F8Qxt-1zR_MY_NeJa?dl=0

Built with python 3.9.2
## Installation:

inside the repository: `pip install -e .`

## Example Inference:

```python
from honeybee_comb_inferer.inference import HoneyBeeCombInferer

model_name = "unet_effnetb0"
path_to_pretrained_models = "path-to-pretrained-models"
device = "cuda:3"

model = HoneyBeeCombInferer(model_name = model_name, path_to_pretrained_models = path_to_pretrained_models, device = device)

path_to_image = "path-to-image"
inferred_mask = model.infer(image = path_to_image)

# for inferring mask without bees
path_to_close_frames = "path-to-close-frames"
inferred_mask_without_bees = model.infer_without_bees(path_to_close_frames)
```

## Example counting of cells:

```python
from honeybee_comb_inferer.cell_counter import CellCounter

cell_counter = CellCounter(inferred_mask=inferred_mask_without_bees, method = "edt")

cell_counter.run_counter()
```

## Running from the command line:
Process can be run via the command line for inferring masks for all images in the specified folder.\
For inferring images located in the default folder ("data\images")
- `python infer.py`. 

To list information about input arguments: 
- `python infer.py -h`
