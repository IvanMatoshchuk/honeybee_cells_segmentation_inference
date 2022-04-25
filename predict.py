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
    parser.add_argument(
        "--source", type=str, help="specify path to folder with image(s)", default="data/images", required=False
    )

    parser.add_argument(
        "--output-path",
        type=str,
        help="specify path to output folder for storing inferred masks",
        default="data/inferred_masks",
        required=False,
    )
    parser.add_argument(
        "--config-path", type=str, help="specify path to config", default="config/config.yaml", required=False
    )
    parser.add_argument(
        "--gpu_num",
        type=int,
        help="select gpu number (default 0)",
        default=0,
        required=False,
        choices=get_gpus_choices(),
    )

    parser.add_argument("--inference", action="store_true", help="run the process of inference")
    parser.add_argument("--gpu", action="store_true", help="use gpu for inference")
    parser.add_argument(
        "-sw",
        "--sliding-window",
        action="store_true",
        help="use sliding window inference, parameters are read from config",
    )

    results = parser.parse_args()
    return results


def main():

    args = get_args()
    print(args)
    return


if __name__ == "__main__":
    main()
