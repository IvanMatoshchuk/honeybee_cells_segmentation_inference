import os
import argparse
from src.utils.utils import get_gpus_choices


def get_args():

    parser = argparse.ArgumentParser(description="Segmentation of Honey Bee Comb")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="choose segmentation model",
        default="unet-effnetb0",
        required=False,
        choices=["unet-effnetb0", "unet-mobilenetv3", "unet-resnet18", "manet-effnetb0"],
    )
    parser.add_argument("--inference", action="store_true", help="run the process of inference")
    parser.add_argument("--gpu", action="store_true", help="use gpu for inference")

    parser.add_argument(
        "--gpu_num",
        type=int,
        help="select gpu number (default 0)",
        default=0,
        required=False,
        choices=get_gpus_choices(),
    )

    results = parser.parse_args()
    return results


def main():

    args = get_args()
    print(args)
    return


if __name__ == "__main__":
    main()
