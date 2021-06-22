import torch as t
import torch.nn.functional as F


class CNN(t.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.backbone = t.nn.Sequential(self.create_block(3, 16, 3, 1, 1, 2),
                                        self.create_block(16, 64, 3, 1, 1, 2),
                                        self.create_block(64, 256, 3, 1, 1, 2),
                                        self.create_block(256, 256, 3, 1, 1, 0),
                                        self.create_block(256, 256, 3, 1, 1, 0))

    def create_block(self, in_channel, out_channel, kernel, stride, padding, pooling):
        return t.nn.Sequential(t.nn.Conv2d(in_channel, out_channel, kernel, stride, padding),
                               t.nn.BatchNorm2d(out_channel),
                               t.nn.GELU(),
                               t.nn.MaxPool2d(pooling))

    def forward(self, x):
        return self.backbone(x)


class Siamese(t.nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        self.backbone = CNN()

    def forward(self, source, target):
        source = self.backbone(source)
        target = self.backbone(target)
        score_map = F.Conv2d(source, target)
        return score_map

