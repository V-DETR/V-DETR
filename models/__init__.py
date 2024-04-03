#Copyright (c) V-DETR authors. All Rights Reserved.
from .model_vdetr import build_vdetr

MODEL_FUNCS = {
    'vdetr': build_vdetr,
}

def build_model(args, dataset_config):
    model = MODEL_FUNCS[args.model_name](args, dataset_config)
    return model