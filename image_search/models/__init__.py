from .resnet import Resnet50


def get_model(model_name, weights_path=None):
    if model_name == 'r50':
        return Resnet50()
    else:
        raise ValueError("model name not supported")
