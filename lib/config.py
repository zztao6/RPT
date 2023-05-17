import os
from argparse import ArgumentParser

BATCHNORM_MOMENTUM = 0.01

class Config(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """
        Defaults
        """
        self.mode = None
        self.device = None
        self.save_path = None
        self.model_path = None
        self.data_path = None
        self.datasize = None
        self.ckpt = None
        self.log_path = None
        self.optimizer = None
        self.bce_loss = None
        self.lr = 1e-5
        self.enc_layer = 2
        self.dec_layer = 2
        self.nepoch = 15
        self.work_dir = None
        self.train_model_name = None
        self.eval = None
        self.config_path = None
        self.cuda_visible_device = None
        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
        self.__dict__.update(self.args)

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description='training code')
        parser.add_argument('-mode', dest='mode', help='predcls/sgcls/sgdet', default='predcls', type=str)
        parser.add_argument('-device', dest='device', default='cuda:0', type=str)
        parser.add_argument('-save_path', default='work_dir/train/', type=str)
        parser.add_argument('-model_path', type=str)
        parser.add_argument('-data_path', default='dataset/ag/', type=str)
        parser.add_argument('-datasize', dest='datasize', help='mini dataset or whole', default='large', type=str)
        parser.add_argument('-ckpt', dest='ckpt', help='checkpoint', default=None, type=str)
        parser.add_argument('-optimizer', help='adamw/adam/sgd', default='adamw', type=str)
        parser.add_argument('-lr', dest='lr', help='learning rate', default=1e-5, type=float)
        parser.add_argument('-nepoch', help='epoch number', default=20, type=float)
        parser.add_argument('-enc_layer', dest='enc_layer', help='spatial encoder layer', default=2, type=int)
        parser.add_argument('-dec_layer', dest='dec_layer', help='temporal decoder layer', default=2, type=int)
        parser.add_argument('-bce_loss', action='store_true')
        parser.add_argument('-log_path', default='work_dir/train/exp.log', dest='log_path', type=str)
        parser.add_argument('-work_dir', default='work_dir/train/', dest='work_dir', type=str)
        parser.add_argument('-train_model_name', default='', dest='train_model_name', type=str)
        parser.add_argument('-cuda_visible_device', default='0', dest='cuda_visible_device', type=str)
        parser.add_argument('-eval', default='semi_constraint', dest='eval', type=str)
        parser.add_argument('-config_path', dest='config_path', type=str)

        return parser
