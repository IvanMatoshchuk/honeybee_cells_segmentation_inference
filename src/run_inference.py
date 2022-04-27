from src.inference.HoneyBeeCombInferer import HoneyBeeCombInferer


def run_inference(args):

    if args.gpu:
        device = "cuda:" + str(args.gpu_num)
    else:
        device = "cpu"

    model = HoneyBeeCombInferer(
        model_name=args.model_name,
        config_path=args.config_path,
        label_classes_path=args.label_classes_path,
        sw_inference=args.sliding_window,
        device=device,
        output_folder_for_masks=args.output_path,
        path_to_pretrained_models=args.models_path,
    )

    model.infer_batch(images_path=args.source)
