"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(torch.autograd.Function):
    r"""
    softmax with cross entropy
    log_softmax:
    y = log(\frac{e^x}{\sum e^{x_k}})
    negative likelyhood:
    z = - \sum t_i y_i, where t is one hot target
    """

    def forward(ctx, x, target, aux_x):
        assert x.dim() == 2, "dimension of input should be 2"

        y = torch.nn.functional.softmax(x, dim=-1)
        aux_y = torch.nn.functional.softmax(aux_x, dim=-1)
        t = torch.nn.functional.one_hot(target, num_classes=x.size(-1))

        output = (-t * torch.log(y)).sum() / y.size(0)

        # ctx.save_for_backward(y, t)
        ctx.save_for_backward(aux_y, t)

        return output

    def backward(ctx, grad_output):
        """
        backward propagation
        """
        y, t = ctx.saved_tensors
        # print((y - t).sum())
        grad_input = grad_output * (y - t) / y.size(0)
        return grad_input, None, None


class Criterion(object):
    def __init__(self):
        self.name = "CosineSimilarity"
        self.criterion = nn.CosineSimilarity(dim=1)

    def __call__(self, pred, target):
        if len(pred) == 2:
            p1, p2 = pred
            if len(target) == 2:
                z1, z2 = target
                val = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5
            else:
                z1, z2, oz1, oz2 = target
                val1 = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5
                val2 = -(self.criterion(p1, oz2.detach()).mean() + self.criterion(p2, oz1.detach()).mean()) * 0.5
                val = (val1 + val2) * 0.5
        else:
            p1, p2, op1, op2 = pred
            if len(target) == 2:
                z1, z2 = target
                val1 = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5
                val2 = -(self.criterion(op1, z2.detach()).mean() + self.criterion(op2, z1.detach()).mean()) * 0.5
                val = (val1 + val2) * 0.5
            else:
                z1, z2, oz1, oz2 = target
                val1 = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5
                val2 = -(self.criterion(op1, z2.detach()).mean() + self.criterion(op2, z1.detach()).mean()) * 0.5
                val3 = -(self.criterion(op1, oz2.detach()).mean() + self.criterion(op2, oz1.detach()).mean()) * 0.5
                val = (val1 + val2 + val3) / 3

        return val


class SimLoss(nn.Module):
    def __init__(self, classes):
        super(SimLoss, self).__init__()
        self.classes = classes

    def forward(self, Sh, y):
        y_onehot = one_hot(y, self.classes)
        Sy = similarity_matrix(y_onehot)
        Sh = similarity_matrix(Sh)
        return F.mse_loss(Sh, Sy)


def similarity_matrix(x):
    ''' Calculate adjusted cosine similarity matrix of size x.size(0) x x.size(0). '''
    no_similarity_std = False
    if x.dim() == 4:
        if not no_similarity_std and x.size(1) > 3 and x.size(2) > 1:
            z = x.view(x.size(0), x.size(1), -1)
            x = z.std(dim=2)
        else:
            x = x.view(x.size(0), -1)
    xc = x - x.mean(dim=1).unsqueeze(1)
    xn = xc / (1e-8 + torch.sqrt(torch.sum(xc ** 2, dim=1))).unsqueeze(1)
    R = xn.matmul(xn.transpose(1, 0)).clamp(-1, 1)
    return R


def one_hot(y, n_dims):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), n_dims).to(y.device)
    return zeros.scatter(scatter_dim, y_tensor, 1)
