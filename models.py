# coding: utf-8
"""
Created on Jun 20, 2018

@author: guoweiyu
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


               
def create_models(opt):
    from networks import SegShuffle
    model = SegShuffle()
    return model


def define_model(opt):
    net = create_models(opt)
    model = _segModel(net)
    use_gpu = len(opt.gpus) > 0
    if use_gpu:
        assert (torch.cuda.is_available())

    if len(opt.gpus) > 0:
        model.cuda()
    return model


class _segModel(nn.Module):
    def __init__(self, model, gpus=[]):
        super(_segModel, self).__init__()
        self.gpu_ids = gpus
        self.model = model

    def forward(self, x):
        if len(self.gpu_ids) > 1 and isinstance(x.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, x, self.gpu_ids)
        else:
            return self.model(x)


class SegmentModel:
    imgs = None
    gts = None
    loss = float("inf")

    def __init__(self, opt, is_train=True):
        self.model = define_model(opt)
        self.save_dir = os.path.join(opt.train_root, opt.model_name)

        self.gpus = opt.gpus
        self.use_gpu = (len(opt.gpus) > 0 and torch.cuda.is_available())

        if is_train or opt.resume:
            which_model = opt.which_model
            self.load_network(opt.model_name, which_model)


        print('---------- Networks initialized -------------')

    def set_test_input(self, imgs):
        self.model.eval()
        if self.use_gpu:
            imgs = imgs.cuda()
        self.imgs = Variable(imgs, requires_grad=False)

    def forward(self):
        return self.model(self.imgs)


    def load_network(self, network_label, record_label):
        save_filename = '%s_%s.pth' % (network_label, record_label)
        save_path = os.path.join(self.save_dir, save_filename)
        print(save_path)
        if os.path.exists(save_path):
            print('loading pretrained model...')
            self.model.load_state_dict(torch.load(save_path))

