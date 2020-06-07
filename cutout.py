# coding=utf8
from __future__ import division
import os
import torch
import torch.utils.data as data
import numpy as np
import pandas
import random
import PIL.Image as Image
from PIL import ImageStat

import albumentations as albu

import pdb



class dataset(data.Dataset):
    def __init__(self, Config, anno, swap_size=[7,7], unswap=None, swap=None, totensor=None, train=False, train_val=False, test=False):
        self.root_path = Config.rawdata_root
        self.numcls = Config.numcls
        self.dataset = Config.dataset
        if isinstance(anno, pandas.core.frame.DataFrame):
            self.paths = anno['ImageName'].tolist()
            self.labels = anno['label'].tolist()
        elif isinstance(anno, dict):
            self.paths = anno['img_name']
            self.labels = anno['label']

        self.cls_dict = get_sample_dict(self.paths, self.labels)
        if train:
            self.update_sample()

        if train_val:
            self.paths, self.labels = random_sample(self.paths, self.labels)
        self.unswap = unswap
        self.swap = swap
        self.totensor = totensor
        self.cfg = Config
        self.train = train
        self.swap_size = swap_size
        self.test = test
        #self.cutout = albu.Compose([albu.CoarseDropout(max_holes=7, max_height=20, max_width=20, p=1)])
        resize_reso = 256
        crop_reso = 224
        self.cutout = albu.Compose([albu.Resize(resize_reso, resize_reso), 
                               albu.Rotate(limit=15),
                               albu.RandomCrop(crop_reso, crop_reso),
                               albu.HorizontalFlip(p=0.5),
                               albu.CoarseDropout(max_holes=7, max_height=20, max_width=20, p=1), 
                               albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                               ])


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_path, self.paths[item])
        img = self.pil_loader(img_path)
        if self.test:
            img = self.totensor(img)
            label = self.labels[item]
            return img, label, self.paths[item]
        img_unswap = self.cutout(image=np.array(img))
        return_img = torch.from_numpy(img_unswap['image']).unsqueeze(0)
        trans_img = return_img.transpose(0, 3).squeeze(3)

        return trans_img, self.labels[item], self.paths[item]
        