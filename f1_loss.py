import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

def f1_loss(predict, target):
    predict = F.softmax(predict, 1)
    batch_size = predict.size(0)
    nb_digits = predict.size(1)
    y = torch.LongTensor(batch_size,1).random_() % nb_digits
    y_onehot = torch.FloatTensor(batch_size, nb_digits)

    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)

    target =  F.softmax(y_onehot.cuda(), 1)
    predict = torch.clamp(predict * (1-target), min=0.01) + predict * target
    tp = predict * target
    tp = tp.sum(dim=0)
    precision = tp / (predict.sum(dim=0) + 1e-8)
    recall = tp / (target.sum(dim=0) + 1e-8)
    f1 = 2 * (precision * recall / (precision + recall + 1e-8))
    return 1 - f1.mean()
