#!/usr/bin/env python

import argparse
import os
import os.path as osp

import torch

import torchfcn

from train_fcn32s import get_log_dir
from train_fcn32s import get_parameters


configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    1: dict(
        max_iteration=100000,
        lr=1.0e-10,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=4000,
        fcn16s_pretrained_model=torchfcn.models.FCN16s.pretrained_model,
    ),
    2: dict(
        max_iteration=100000,
        lr=1.0e-4,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=4000,
    )
}


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=str, required=True)
    parser.add_argument('-c', '--config', type=int, default=1,
                        choices=configurations.keys())
    parser.add_argument('--resume', help='Checkpoint path')
    args = parser.parse_args()

    gpu = args.gpu
    cfg = configurations[args.config]
    out = get_log_dir('fcn8s-atonce', args.config, cfg)
    resume = args.resume

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    if torch.cuda.device_count() == 1:
        batch_size = 1
    else:
        batch_size = 2 * torch.cuda.device_count()

    # 1. dataset

    root = osp.expanduser('~/data/datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    mix_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.CityScapesClassSeg(
            root, split=['train', 'val'], transform=True, preprocess=False,
        ), batch_size=batch_size, shuffle=True, **kwargs
    )
    train_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.CityScapesClassSeg(
            root, split=['train'], transform=True, preprocess=False,
        ), batch_size=batch_size, shuffle=True, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.CityScapesClassSeg(
            root, split=['val'], transform=True, preprocess=False,
        ), batch_size=batch_size, shuffle=False, **kwargs
    )

    # train_loader = torch.utils.data.DataLoader(
    #     torchfcn.datasets.SBDClassSeg(root, split='train', transform=True),
    #     batch_size=1, shuffle=True, **kwargs)
    # val_loader = torch.utils.data.DataLoader(
    #     torchfcn.datasets.VOC2011ClassSeg(
    #         root, split='seg11valid', transform=True),
    #     batch_size=1, shuffle=False, **kwargs)

    # 2. model

    model = torchfcn.models.FCN8sAtOnce(n_class=20)

    start_epoch = 0
    start_iteration = 0
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        vgg16 = torchfcn.models.VGG16(pretrained=True)
        model.copy_params_from_vgg16(vgg16)
    if cuda:
        if torch.cuda.device_count() == 1:
            model = model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # 3. optimizer

    optim = torch.optim.Adam(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True),
             'lr': cfg['lr'] * 2, 'weight_decay': 0},
        ],
        lr=cfg['lr'],
        # momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'])
    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = torchfcn.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        mix_loader=mix_loader,
        out=out,
        nEpochs=5,
        max_iter=cfg['max_iteration'],
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
