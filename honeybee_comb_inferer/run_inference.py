from honeybee_comb_inferer.inference.HoneyBeeCombInferer import HoneyBeeCombInferer


def run_inference(args):

    if args.gpu:
        device = "cuda:" + str(args.gpu_num)
    else:
        device = "cpu"

    model = HoneyBeeCombInferer(
        model_name=args.model_name,
        path_to_pretrained_models=args.models_path,
        config=args.config_path,
        label_classes_config=args.label_classes_path,
        sw_inference=args.sliding_window,
        device=device,
    )

    model.infer_batch(images_path=args.source)
