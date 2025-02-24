from .resnet_2d3d import *

"""
most of the codes adapted from original repo https://github.com/TengdaHan/MemDPC
"""


def select_resnet(
    network,
):
    param = {"feature_size": 1024}
    if network == "resnet18":
        model = resnet18_2d3d_full(track_running_stats=True)
        param["feature_size"] = 256
    elif network == "resnet34":
        model = resnet34_2d3d_full(track_running_stats=True)
        param["feature_size"] = 256
    elif network == "resnet50":
        model = resnet50_2d3d_full(track_running_stats=True)

    elif network == "resnet18_simplified":
        model = resnet18_2d3d_simplified(track_running_stats=True)
        param["feature_size"] = 128
    else:
        raise NotImplementedError

    return model, param
