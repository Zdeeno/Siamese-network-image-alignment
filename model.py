import random
import torch
import torch as t
from torch.nn.functional import conv2d, conv1d
import torch.nn as nn
import math
from torch.nn import functional as F
from copy import deepcopy
import os
import errno


def create_conv_block(in_channel, out_channel, kernel, stride, padding, pooling, bn=True, relu=True,
                      pool_layer=False):
    net_list = [t.nn.Conv2d(in_channel, out_channel, kernel, stride, padding, padding_mode="circular")]
    if bn:
        net_list.append(t.nn.BatchNorm2d(out_channel))
    if relu:
        net_list.append(t.nn.ReLU())
    if pooling[0] > 0 or pooling[1] > 0:
        if not pool_layer:
            net_list.append(t.nn.MaxPool2d(pooling, pooling))
        else:
            net_list.append(t.nn.Conv2d(out_channel, out_channel, pooling, pooling))
    return t.nn.Sequential(*net_list)


def create_residual_block(fs, ch, pad):
    return t.nn.Sequential(create_conv_block(ch, ch, fs, 1, pad, (0, 0), bn=True, relu=True),
                           t.nn.ReLU(),
                           create_conv_block(ch, ch, fs, 1, pad, (0, 0), bn=False, relu=False))


class CNNOLD(t.nn.Module):

    def __init__(self, lp, fs, ech, res, legacy=True):
        super(CNNOLD, self).__init__()
        pad = fs // 2
        self.res = res
        self.legacy = legacy
        if self.legacy:
            self.backbone = t.nn.Sequential(*[create_conv_block(3, 16, 3, 1, 1, (2, 2), pool_layer=False),
                                              create_conv_block(16, 64, 3, 1, 1, (2, 2), pool_layer=False),
                                              create_conv_block(64, 256, 3, 1, 1, (2, 2), pool_layer=False),
                                              create_conv_block(256, 512, 3, 1, 1, (2, 1), pool_layer=False),
                                              create_conv_block(512, 256, 3, 1, 1, (3, 1), bn=False, relu=False,
                                                                pool_layer=False)])
        else:
            self.l1 = create_conv_block(3, 16, fs, 1, pad, (2, 2), pool_layer=lp, relu=False)
            self.l2 = create_conv_block(16, 64, fs, 1, pad, (2, 2), pool_layer=lp, relu=False)
            self.l3 = create_conv_block(64, 256, fs, 1, pad, (2, 2), pool_layer=lp, relu=False)
            self.l4 = create_conv_block(256, 512, fs, 1, pad, (2, 1), pool_layer=lp, relu=False)
            self.l5 = create_conv_block(512, ech, fs, 1, pad, (3, 1), relu=False, pool_layer=lp)
            if self.res == 1:
                # residual blocks
                self.l1res = create_residual_block(fs, 16, pad)
                self.l2res = create_residual_block(fs, 64, pad)
                self.l3res = create_residual_block(fs, 256, pad)
                self.l4res = create_residual_block(fs, 512, pad)
            if self.res == 2:
                # Squeeze and excitation blocks
                self.l1res = SE_Block(16)
                self.l2res = SE_Block(64)
                self.l3res = SE_Block(256)
                self.l4res = SE_Block(512)
            if self.res == 3:
                self.l1res = t.nn.Sequential(create_residual_block(fs, 16, pad), SE_Block(16))
                self.l2res = t.nn.Sequential(create_residual_block(fs, 64, pad), SE_Block(64))
                self.l3res = t.nn.Sequential(create_residual_block(fs, 256, pad), SE_Block(256))
                self.l4res = t.nn.Sequential(create_residual_block(fs, 512, pad), SE_Block(512))

    def forward(self, x):
        if self.legacy:
            return self.backbone(x)
        else:
            if self.res > 0:
                x_tmp = self.l1(x)
                x = self.l1res(x_tmp) + x_tmp
                x = t.relu(x)
                x_tmp = self.l2(x)
                x = self.l2res(x_tmp) + x_tmp
                x = t.relu(x)
                x_tmp = self.l3(x)
                x = self.l3res(x_tmp) + x_tmp
                x = t.relu(x)
                x_tmp = self.l4(x)
                x = self.l4res(x_tmp) + x_tmp
                x = t.relu(x)
                x = self.l5(x)
            else:
                x = self.l1(x)
                x = t.relu(x)
                x = self.l2(x)
                x = t.relu(x)
                x = self.l3(x)
                x = t.relu(x)
                x = self.l4(x)
                x = t.relu(x)
                x = self.l5(x)
            return x


def get_custom_CNN(lp, fs, ech, res):
    return CNNOLD(lp, fs, ech, res)


class SE_Block(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class Siamese(t.nn.Module):

    def __init__(self, backbone, padding=3):
        super(Siamese, self).__init__()
        self.backbone = backbone
        self.padding = padding
        # self.wb = t.nn.Parameter(t.tensor([1.0, 0.0]), requires_grad=True)
        self.out_batchnorm = t.nn.BatchNorm2d(1)

    def forward(self, source, target, padding=None, displac=None):
        source = self.backbone(source)
        target = self.backbone(target)
        if displac is None:
            # regular walk through
            score_map = self.match_corr(target, source, padding=padding)
            score_map = self.out_batchnorm(score_map)
            return score_map.squeeze(1).squeeze(1)
        else:
            # for importance visualisation
            shifted_target = t.roll(target, displac, -1)
            score = source * shifted_target
            score = t.sum(score, dim=[1, 2])
            return score

    def match_corr(self, embed_ref, embed_srch, padding=None):
        """ Matches the two embeddings using the correlation layer. As per usual
        it expects input tensors of the form [B, C, H, W].
        Args:
            embed_ref: (torch.Tensor) The embedding of the reference image, or
                the template of reference (the average of many embeddings for
                example).
            embed_srch: (torch.Tensor) The embedding of the search image.
        Returns:
            match_map: (torch.Tensor) The correlation between
        """
        if padding is None:
            padding = self.padding
        b, c, h, w = embed_srch.shape
        _, _, h_ref, w_ref = embed_ref.shape
        # Here the correlation layer is implemented using a trick with the
        # conv2d function using groups in order to do the correlation with
        # batch dimension. Basically we concatenate each element of the batch
        # in the channel dimension for the search image (making it
        # [1 x (B.C) x H' x W']) and setting the number of groups to the size of
        # the batch. This grouped convolution/correlation is equivalent to a
        # correlation between the two images, though it is not obvious.

        if self.training:
            match_map = conv2d(embed_srch.view(1, b * c, h, w), embed_ref, groups=b, padding=(0, padding))
            match_map = match_map.permute(1, 0, 2, 3)
        else:
            match_map = F.conv2d(F.pad(embed_srch.view(1, b * c, h, w), pad=(padding, padding, 1, 1), mode='circular'),
                                 embed_ref, groups=b)

            match_map = t.max(match_map.permute(1, 0, 2, 3), dim=2)[0].unsqueeze(2)
        return match_map


def get_parametrized_model(lp, fs, ech, res, pad, device):
    backbone = get_custom_CNN(lp, fs, ech, res)
    model = Siamese(backbone, padding=pad).to(device)
    return model


def save_model(model, name, epoch, optimizer=None):
    t.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None
    }, "./results_" + name + "/model_" + str(epoch) + ".pt")
    print("Model saved to: " + "./results_" + name + "/model_" + str(epoch) + ".pt")


def load_model(model, path, optimizer=None):
    checkpoint = t.load(path, map_location=t.device("cpu"))
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded model at", path)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer
    else:
        return model


def jit_save(model, name, epoch, arb_in, args):
    # save model arguments
    filename = "./results_" + name + "/model.info"
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, "w") as f:
        f.write(str(args))

    # save actual model
    t.jit.save(t.jit.trace(model, arb_in), "./results_" + name + "/model_" + str(epoch) + ".jit")


def jit_load(path, device):
    return t.jit.load(path, map_location=device)
