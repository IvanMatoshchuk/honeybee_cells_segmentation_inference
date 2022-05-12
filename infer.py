import os
import argparse
import logging
from honeybee_comb_inferer.utils.utils import get_gpus_choices
from honeybee_comb_inferer.run_inference import run_inference


logging.basicConfig(format="%(name)s:%(levelname)s:%(message)s ")
log = logging.getLogger(name=os.path.basename(__file__))
log.setLevel(logging.INFO)


def get_args():

    parser = argparse.ArgumentParser(
        description="Segmentation of Honey Bee Comb",
        epilog="By default processes all images in 'data/images' folder and outputs in 'data/inferred_masks'",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        help="choose segmentation model",
        default="unet_effnetb0",
        required=False,
        choices=["unet_effnetb0", "unet_mobilenetv3", "unet_resnet18", "manet_effnetb0"],
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
        "--models-path",
        type=str,
        help="specify path to pretrained models' state-dicts",
        default="models",
        required=False,
    )
    parser.add_argument(
        "--config-path", type=str, help="specify path to config.yaml", default="config/config.yaml", required=False
    )

    parser.add_argument(
        "--label-classes-path",
        type=str,
        help="specify path to label_classes.json (from hasty.ai)",
        default="data/label_classes.json",
        required=False,
    )

    parser.add_argument(
        "--gpu_num",
        type=int,
        help="select gpu number (default 0)",
        default=0,
        required=False,
        choices=get_gpus_choices(),
    )

    parser.add_argument("-g", "--gpu", action="store_true", help="use gpu for inference")
    parser.add_argument(
        "-sw",
        "--sliding-window",
        action="store_true",
        help="use sliding window inference, parameters are read from config",
    )

    results = parser.parse_args()
    return results


def check_n_images(path: str) -> int:

    return len(os.listdir(path))


def main() -> None:

    args = get_args()
    log.info(args)

    log.info(f"Images in folder: {check_n_images(args.source)}")

    run_inference(args)

    return None


if __name__ == "__main__":
    main()
