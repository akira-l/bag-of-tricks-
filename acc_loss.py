#coding=utf8
from __future__ import print_function, division

import os,time,datetime
import numpy as np
from math import ceil
import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

#from torchvision.utils import make_grid, save_image

from utils.utils import LossRecord, clip_gradient
from utils.eval_model import eval_turn
from models.focal_loss import FocalLoss
from utils.Asoftmax_loss import AngleLoss

from dataset.dataset_DCL import collate_fn4train, collate_fn4val, collate_fn4test, collate_fn4backbone, dataset
from config import LoadConfig, load_data_transformers

from models.BiTempered_module import BiTemperedLayer
from models.BiTemperedLoss import BiTemperedLoss 
 

import pdb




def acc_loss(y_true, y_pred):
    # y_pred = y_pred.round()
    tp = (y_pred*y_true).sum(1)
    fp = ((1-y_true)*y_pred).sum(1)
    acc = tp/(tp+fp)
    return 1-acc.mean()


def train(Config,
          model,
          epoch_num,
          start_epoch,
          optimizer,
          exp_lr_scheduler,
          data_loader,
          save_dir,
          data_ver='all',
          data_size=448,
          savepoint=500,
          checkpoint=1000
          ):


    step = 0
    eval_train_flag = False
    rec_loss = []
    checkpoint_list = []

    train_batch_size = data_loader['train'].batch_size
    train_epoch_step = data_loader['train'].__len__()
    train_loss_recorder = LossRecord(train_batch_size)
    bitempered_layer = BiTemperedLayer(t1=0.9, t2=1.05)
    bitempered_loss = BiTemperedLoss()

    add_loss = nn.L1Loss()
    get_focal_loss = FocalLoss()
    get_angle_loss = AngleLoss()
    get_ce_loss = nn.CrossEntropyLoss()
    get_l2_loss = nn.MSELoss()

    for epoch in range(start_epoch,epoch_num-1):
        exp_lr_scheduler.step(epoch)
        model.train(True)

        save_grad = []

        for batch_cnt, data in enumerate(data_loader['train']):
            step += 1
            loss = 0
            model.train(True)

            inputs, labels, img_names = data
            inputs = inputs.cuda()
            labels = torch.from_numpy(np.array(labels)).cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            # ce_loss = get_ce_loss(outputs, labels)
            labels_onehot = torch.zeros(outputs.shape[0], 50030).cuda().scatter_(1, labels.unsqueeze_(1), 1)
            ce_loss = acc_loss(labels_onehot, F.softmax(outputs, 1))
            loss += ce_loss

            loss.backward()
            #torch.cuda.synchronize()

            optimizer.step()
            #torch.cuda.synchronize()




    log_file.close()


