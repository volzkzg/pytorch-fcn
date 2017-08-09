#!/usr/bin/env python
import os
import collections
import os.path as osp

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data

class CityScapesClassSegBase(data.Dataset):

    class_names = np.array([
        'road',
        'sidewalk',
        'parking',
        'rail track',
        'person',
        'rider',
        'car',
        'truck',
        'bus',
        'on rails',
        'motorcycle',
        'bicycle',
        'caravan',
        'trailer',
        'building',
        'wall',
        'fence',
        'guard rail',
        'bridge',
        'tunnel',
        'pole',
        'pole group',
        'traffic sign',
        'traffic light',
        'vegetation',
        'terrain',
        'sky',
        'ground',
        'dynamic',
        'static',
    ])

    def __init__(self, root, split='train', transform=False, preprocess=False):
        self.root = root
        self.split = split
        self._transform = transform

        dataset_dir = osp.join(self.root, 'cityscapes/')

        if (preprocess):
            self.preprocess_dataset(self, dataset_dir, split=split)

        tar_img_dir = osp.join(dataset_dir, 'leftImg8bit/%s' % split)
        tar_lbl_dir = osp.join(dataset_dir, 'gtFine/%s' % split)

        for city in os.listdir(tar_img_dir):
            city_img_dir = osp.join(tar_img_dir, city)
            city_lbl_dir = osp.join(tar_lbl_dir, city)
            imgsets_file = osp.join(city_img_dir, 'imgsets.txt')

            if not osp.isdir(city_img_dir):
                continue

            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(city_img_dir, '%s_leftImg8bit.png' % did)
                lbl_file = osp.join(
                    city_lbl_dir, '%s_gtFine_color.png' % did
                )
                self.files.append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data_file = self.files[index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1
        if self._transform:
            # return self.transform(img, lbl)
            return img, lbl
        else:
            return img, lbl

    def preprocess_dataset(self, dataset_dir, split):
        tar_img_dir = osp.join(dataset_dir, 'leftImg8bit/%s' % split)
        tar_lbl_dir = osp.join(dataset_dir, 'gtFine/%s' % split)

        for city in os.listdir(tar_img_dir):
            city_dir = osp.join(tar_img_dir, city)
            if osp.isdir(city_dir):
                os.system('cd %s' % city_dir)
                os.system('mogrify -format jpg -resize 1024*512 *.png')

                # make imgsets file
                with open(
                        osp.join(city_dir, 'imgsets.txt'), 'w'
                ) as imgsets_file:
                    for img_name in os.listdir(city_dir):
                        img_name, _ = img_name.split('_leftImg8bit')
                        imgsets_file.write(img_name + '\n')

        for city in os.listdir(tar_lbl_dir):
            city_dir = osp.join(tar_lbl_dir, city)
            if osp.isdir(city_dir):
                os.system('cd %s' % city_dir)
                os.system('mogrify -resize 1024*512 *.png')
