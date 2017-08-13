#!/usr/bin/env python
import os
import collections
import os.path as osp

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data


class CityScapesClassSeg(data.Dataset):

    mean_bgr = np.array((72.78044, 83.21195, 73.45286), dtype=np.float32)
    full_to_train = {
        -1: 19, 0: 19, 1: 19, 2: 19, 3: 19, 4: 19, 5: 19, 6: 19,
        7: 0, 8: 1, 9: 19, 10: 19, 11: 2, 12: 3, 13: 4, 14: 19,
        15: 19, 16: 19, 17: 5, 18: 19, 19: 6, 20: 7, 21: 8, 22: 9,
        23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 19,
        30: 19, 31: 16, 32: 17, 33: 18
    }
    train_to_full = {
        0: 7, 1: 8, 2: 11, 3: 12, 4: 13, 5: 17, 6: 19, 7: 20, 8: 21, 9: 22,
        10: 23, 11: 24, 12: 25, 13: 26, 14: 27, 15: 28, 16: 31, 17: 32, 18: 33,
        19: 0
    }
    class_names = np.array([
        'Unlabeled',
        'Road',
        'Sidewalk',
        'Building',
        'Wall',
        'Fence',
        'Pole',
        'TrafficLight',
        'TrafficSign',
        'Vegetation',
        'Terrain',
        'Sky',
        'Person',
        'Rider',
        'Car',
        'Truck',
        'Bus',
        'Train',
        'Motorcycle',
        'Bicycle'
    ])

    # https://github.com/e-lab/ENet-training/blob/master/train/data/loadCityscape.lua
    class_map = {
        -1: 1,  # Licence plate
        0: 1,  # Unlabeled
        1: 1,  # Ego vehicle
        2: 1,  # Rectification border
        3: 1,  # Out of roi
        4: 1,  # Static
        5: 1,  # Dynamic
        6: 1,  # Ground
        7: 2,  # Road
        8: 3,  # Sidewalk
        9: 1,  # Parking
        10: 1,  # Rail track
        11: 4,  # Building
        12: 5,  # Wall
        13: 6,  # Fence
        14: 1,  # Guard rail
        15: 1,  # Bridge
        16: 1,  # Tunnel
        17: 7,  # Pole
        18: 1,  # Polegroup
        19: 8,  # TrafficLight
        20: 9,  # TrafficSign
        21: 10,  # Vegetation
        22: 11,  # Terrain
        23: 12,  # Sky
        24: 13,  # Person
        25: 14,  # Rider
        26: 15,  # Car
        27: 16,  # Truck
        28: 17,  # Bus
        29: 1,  # Caravan
        30: 1,  # Trailer
        31: 18,  # Train
        32: 19,  # Motorcycle
        33: 20,  # Bicycle
    }

    def __init__(self, root, split='train', transform=False, preprocess=False):
        self.root = root
        self.split = split
        self._transform = transform
        self.files = []

        dataset_dir = osp.join(self.root, 'cityscapes/')

        if (preprocess):
            self.preprocess_dataset(dataset_dir, split)

        tar_img_dir = osp.join(dataset_dir, 'leftImg8bit/%s' % split)
        tar_lbl_dir = osp.join(dataset_dir, 'gtFine/%s' % split)  # gtFine

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
                    city_lbl_dir, '%s_gtFine_labelIds.png' % did
                )
                self.files.append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def myfunc(self, a):
        # print(a)
        if a in self.full_to_train:
            return self.full_to_train[a]
        else:
            19

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

        # print(str(w) + ' ' + str(h))
        w, h = lbl.shape
        lbl = lbl.reshape(-1)
        vfunc = np.vectorize(self.myfunc)
        lbl = vfunc(lbl).reshape(w, h)

        # for i in range(lbl.shape[0]):
        #     for j in range(lbl.shape[1]):
        #         if lbl[i][j] in self.full_to_train:
        #             lbl[i][j] = self.full_to_train[lbl[i][j]]
        #         else:
        #             lbl[i][j] = 19

        # tmp = lbl.copy()
        # for k, v in self.full_to_train.items():
        #     tmp[lbl == k] = v

        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        # img.div(255)
        # img[0].add_(-0.485).div_(0.229)
        # img[1].add_(-0.456).div_(0.224)
        # img[2].add_(-0.406).div_(0.225)
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl

    def preprocess_dataset(self, dataset_dir, split):
        tar_img_dir = osp.join(dataset_dir, 'leftImg8bit/%s' % split)
        tar_lbl_dir = osp.join(dataset_dir, 'gtFine/%s' % split)

        for city in os.listdir(tar_img_dir):
            city_dir = osp.join(tar_img_dir, city)
            if osp.isdir(city_dir):
                print(city_dir)
                os.chdir(city_dir)
                os.system('mogrify -format jpg -resize 1024x512 *.png')

                # make imgsets file
                with open(
                        osp.join(city_dir, 'imgsets.txt'), 'w'
                ) as imgsets_file:
                    for img_name in os.listdir(city_dir):
                        if '_leftImg8bit' not in img_name:
                            continue
                        img_name, _ = img_name.split('_leftImg8bit')
                        imgsets_file.write(img_name + '\n')

        for city in os.listdir(tar_lbl_dir):
            city_dir = osp.join(tar_lbl_dir, city)
            if osp.isdir(city_dir):
                os.chdir(city_dir)
                os.system('mogrify -resize 1024x512 *.png')
