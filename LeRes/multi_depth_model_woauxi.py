from LeRes import network_auxi as network
from LeRes.net_tools import get_func
import torch
import torch.nn as nn
from collections import OrderedDict


def strip_prefix_if_present(state_dict=None, prefix="module."):
    if state_dict is None:
        depth_dict='./res101.pth'
        depth_dict = torch.load(depth_dict)
        state_dict = depth_dict['depth_model']
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

class RelDepthModel(nn.Module):
    def __init__(self, backbone='resnet50'):
        super(RelDepthModel, self).__init__()
        if backbone == 'resnet50':
            encoder = 'resnet50_stride32'
        elif backbone == 'resnext101':
            encoder = 'resnext101_stride32x8d'
        self.depth_model = DepthModel(encoder)

    def inference(self, rgb):
        with torch.no_grad():
            input = rgb.cuda()
            depth = self.depth_model(input)
            pred_depth_out = depth - depth.min() + 0.01
            return pred_depth_out
        
    def check_feature(self, rgb):
        with torch.no_grad():
            input = rgb.cuda()
            feature = self.depth_model(input)
            return feature


class DepthModel(nn.Module):
    def __init__(self, encoder):
        super(DepthModel, self).__init__()
        backbone = network.__name__.split('.')[-1] + '.' + encoder
        self.encoder_modules = get_func(backbone)()
        self.decoder_modules = network.Decoder()

    def forward(self, x):
        lateral_out = self.encoder_modules(x)
        out_logit = self.decoder_modules(lateral_out)
        return out_logit
        # return lateral_out