import random
import torch as t
from torch.nn.functional import conv2d, conv1d
import torch.nn as nn
import math
from einops import rearrange
from torch.nn import functional as F


class CNN(t.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.backbone = t.nn.Sequential(self._create_block(3, 16, 3, 1, 1, (2, 2)),
                                        self._create_block(16, 64, 3, 1, 1, (2, 2)),
                                        self._create_block(64, 256, 3, 1, 1, (2, 2)),
                                        self._create_block(256, 512, 3, 1, 1, (2, 1)),
                                        self._create_block(512, 128, 3, 1, 1, (3, 1)))    # 128 channels out feels better

    def _create_block(self, in_channel, out_channel, kernel, stride, padding, pooling):
        net_list = [t.nn.Conv2d(in_channel, out_channel, kernel, stride, padding),
                    t.nn.BatchNorm2d(out_channel),
                    t.nn.ReLU()]
        if pooling[0] > 0 or pooling[1] > 0:
            net_list.append(t.nn.MaxPool2d(pooling, pooling))
        return t.nn.Sequential(*net_list)

    def forward(self, x):
        x = self.backbone(x)
        return x


class VGG11EmbeddingNet_5c(nn.Module):
    """ Embedding branch based on Pytorch's VGG11 with Batchnorm (https://pytor
    ch.org/docs/stable/torchvision/models.html). This is version 5c, meaning
    that it has 5 convolutional layers, it follows the original model up until
    the 13th layer (The ReLU after the 4th convolution), in order to keep the
    total stride equal to 4. It adds the 5th convolutional layer which acts as
    a bottleck a feature bottleneck reducing the features from 256 to 32 and
    must always be trained. The layers 0 to 13 can be loaded from
    torchvision.models.vgg11_bn(pretrained=True)
    """

    def __init__(self):
        super(VGG11EmbeddingNet_5c, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3,
                                                  stride=1, bias=True),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2, stride=2),

                                        nn.Conv2d(64, 128, kernel_size=3,
                                                  stride=1, bias=True),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2, stride=2),
                                        nn.Conv2d(128, 256, kernel_size=3,
                                                  stride=1, bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.Conv2d(256, 256, kernel_size=3,
                                                  stride=1, bias=True),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        # Added ConvLayer, not in original model
                                        nn.Conv2d(256, 32, kernel_size=3,
                                                  stride=1, bias=True))

    def forward(self, x):
        output = self.features(x)
        return output


def get_pretrained_VGG11():
    from torchvision.models.vgg import vgg11_bn
    pretrained_dict = vgg11_bn(pretrained=True).state_dict()
    model = VGG11EmbeddingNet_5c()
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model


def get_custom_CNN():
    return CNN()


class Siamese(t.nn.Module):

    def __init__(self, backbone, padding=3):
        super(Siamese, self).__init__()
        self.backbone = backbone
        self.out_batchnorm = t.nn.BatchNorm2d(1)
        self.padding = padding

    def forward(self, source, target, padding=None):
        source = self.backbone(source)
        target = self.backbone(target)
        score_map = self.match_corr(target, source, padding=padding)
        score_map = self.out_batchnorm(score_map)
        return score_map.squeeze(1).squeeze(1)

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
        match_map = conv2d(embed_srch.view(1, b * c, h, w),
                           embed_ref, groups=b, padding=(0, padding))
        # Here we reorder the dimensions to get back the batch dimension.
        match_map = match_map.permute(1, 0, 2, 3)

        return match_map


# TRANSFORMER MODEL ------------------------------------------


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = t.zeros(max_len, d_model)
        position = t.arange(0, max_len, dtype=t.float).unsqueeze(1)
        div_term = t.exp(t.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = t.sin(position * div_term)
        pe[:, 1::2] = t.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, rnd=True):
        if self.training and rnd:
            shift = random.randint(0, 16)
        else:
            shift = 0
        x = x + self.pe[shift:x.size(0)+shift, :]
        return self.dropout(x)


class Transformer(t.nn.Module):
    def __init__(self, backbone, d_model, num_layers, n_head, dim, dropout=0.1):
        super(Transformer, self).__init__()
        self.backbone = backbone
        self.pe = PositionalEncoding(d_model)
        self.transformer = t.nn.Transformer(d_model, n_head, num_layers, num_layers, dim, dropout, "gelu")
        self.out = t.nn.Linear(d_model, 1)

    def forward(self, source, target):
        source = self.backbone(source)
        target = self.backbone(target)
        # (B, CH, H, W) -> (W, B, CHxH)
        source = source.transpose(1, 3).transpose(0, 1).flatten(2, 3)
        target = target.transpose(1, 3).transpose(0, 1).flatten(2, 3)
        target = self.pe(target, rnd=False)
        source = self.pe(source, rnd=True)
        dec_out = self.transformer(target, source)
        dec_out = self.out(dec_out).squeeze(-1)
        out = dec_out.flatten(-1).transpose(0, 1)
        print(source.size(), target.size(), out.size())
        return out  # (B, W)


def save_model(model, name, epoch, optimizer=None):
    t.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None
    }, "./results_" + name + "/model_" + str(epoch) + ".pt")
    print("Model saved to: " + "./results_" + name + "/model_" + str(epoch) + ".pt")


def load_model(model, path, optimizer=None):
    checkpoint = t.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded model at", path)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer
    else:
        return model

