import torch as t
from torch.nn.functional import conv2d, conv1d


class CNN(t.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.backbone = t.nn.Sequential(self._create_block(3, 16, 3, 1, 1, 2),
                                        self._create_block(16, 64, 3, 1, 1, 2),
                                        self._create_block(64, 256, 3, 1, 1, 2),
                                        self._create_block(256, 256, 3, 1, 1, 0),
                                        self._create_block(256, 256, 3, 1, 1, 0))

    def _create_block(self, in_channel, out_channel, kernel, stride, padding, pooling):
        net_list = [t.nn.Conv2d(in_channel, out_channel, kernel, stride, padding),
                    t.nn.BatchNorm2d(out_channel),
                    t.nn.GELU()]
        if pooling > 0:
            net_list.append(t.nn.MaxPool2d(pooling, pooling))
        return t.nn.Sequential(*net_list)

    def forward(self, x):
        return self.backbone(x)


class Siamese(t.nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        self.backbone = CNN()
        self.out_batchnorm = t.nn.BatchNorm2d(1)

    def forward(self, source, target):
        source = self.backbone(source)
        target = self.backbone(target)
        score_map = self.match_corr(target, source)
        score_map = self.out_batchnorm(score_map)
        return score_map.squeeze(1).squeeze(1)

    def match_corr(self, embed_ref, embed_srch):
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
                             embed_ref, groups=b, padding=(0, 3))
        # Here we reorder the dimensions to get back the batch dimension.
        match_map = match_map.permute(1, 0, 2, 3)
        return match_map

