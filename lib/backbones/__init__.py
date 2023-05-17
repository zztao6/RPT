import torch
import torch.nn as nn

from .slowfast import *
# from utils import load_pretrain
from lib.backbones.utils import load_pretrain

def model_entry(config):
    return globals()[config['arch']](**config['kwargs'])


class AVA_backbone(nn.Module):
    def __init__(self, config):
        super(AVA_backbone, self).__init__()
        
        self.config = config
        self.module = model_entry(config)
        
        if config.get('pretrain', None) is not None:
            load_pretrain(config.pretrain, self.module)
                
        if not config.get('learnable', True):
            self.module.requires_grad_(False)

    # data: clips
    # returns: features
    def forward(self, clip_batch):
        N, T, C, H, W = clip_batch.shape
        clip_batch = clip_batch.reshape([N, C, T, H, W])
        clip_output = self.module(clip_batch)

        return {'features': clip_output}
