from src.inference.HoneyBeeCombInferer import HoneyBeeCombInferer


def run_inference(args):

    model = HoneyBeeCombInferer(model_name=args.model_name)

    model.infer_batch(images_path=args.source)
