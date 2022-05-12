# honey_bee_segmentation_inference
Inference pipeline for performing segmentation of honey bee comb.
Pipeline can be used for infering all images in the specified folder (execution via command line) or used as a package in python scripts.

## Installation:

inside the repository: `pip install -e .`

## Example:

```python
from honeybee_comb_inferer.inference.HoneyBeeCombInferer import HoneyBeeCombInferer

model_name = "unet_effnetb0"
device = "cuda:3"

path_to_image = "honeybee_cells_segmentation_inference/data/images/Cam_0_2019-07-24T15_29_46.791050+00_00.png"

model = HoneyBeeCombInferer(model_name = model_name, device = device)

pred = model.infer(image = path_to_image)
print(pred.shape)

```
Cab be used via command line:
Infering masks for all images in the specified folder:
- `python infer.py`. To list information about input arguments run `python infer.py -h`



### Setup ###
Built with python 3.9.2

* from project directory in terminal `pip install -r requirements.txt`

### Usage ###



Usage as a package in python script:
- `from .. import ..`


TBD


