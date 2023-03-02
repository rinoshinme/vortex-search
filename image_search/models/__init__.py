from .resnet import Resnet50Extractor
from .clip import CLIPExtractor


def get_model(model_type, model_name, weights_path=None):
    error_message = 'model not supported'

    if model_type.lower() == 'resnet':
        if model_name == 'r50':
            return Resnet50Extractor()
        else:
            raise ValueError(error_message)
    elif model_type.lower() == 'clip':
        if model_name in ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'RN50', 'RN101']:
            return CLIPExtractor(model_name)
        else:
            raise ValueError(error_message)

    else:
        raise ValueError(error_message)
