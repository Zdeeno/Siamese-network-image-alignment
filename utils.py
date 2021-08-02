from typing import Dict, Optional

import matplotlib.pyplot as plt
import torch
import torch as t
from torch.nn import functional as F
import numpy as np
from pathlib import Path
import kornia as K
from styleaug import StyleAugmentor
import random


class GANAugemntation(t.nn.Module):

    def __init__(self, p=0.5):
        super(GANAugemntation, self).__init__()
        self.augmentor = StyleAugmentor()
        self.p = p

    def forward(self, x):
        with torch.no_grad():
            if random.random() < self.p and x.size(-1) > 40:
                return self.augmentor(x, alpha=0.25)
            else:
                return x


AUG_P = 0.2

batch_augmentations = t.nn.Sequential(
    # GANAugemntation(p=0.2),
    K.augmentation.RandomAffine(t.tensor(10.0),
                                t.tensor([16 / 512, 0.25]),
                                align_corners=False, p=0.3),
    K.augmentation.RandomBoxBlur(p=AUG_P),
    K.augmentation.RandomChannelShuffle(p=AUG_P),
    K.augmentation.RandomPerspective(distortion_scale=0.1, p=AUG_P),
    # K.augmentation.RandomPosterize(p=0.2),    CPU only
    K.augmentation.RandomSharpness(p=AUG_P),
    K.augmentation.RandomSolarize(p=AUG_P),
    K.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=AUG_P),
    K.augmentation.RandomGaussianNoise(std=0.15, p=AUG_P),
    K.augmentation.RandomElasticTransform(p=AUG_P),
    # K.augmentation.RandomEqualize(p=0.2),     CPU only
    K.augmentation.RandomGrayscale(p=AUG_P)
)


def plot_samples(source, target, heatmap, prediction=None, name=0, dir="results/0/"):
    if prediction is None:
        f, axarr = plt.subplots(2)
        target_fullsize = t.zeros_like(source)
        target_width = target.size(-1)
        source_width = source.size(-1)
        heatmap_width = heatmap.size(-1)
        heatmap_idx = t.argmax(heatmap)
        target_fullsize_start = int(heatmap_idx * (source_width/heatmap_width) - target_width//2)
        target_fullsize[:, :, max(target_fullsize_start, 0):target_fullsize_start+target_width] = target
        axarr[0].imshow(source.permute(1, 2, 0))
        axarr[1].imshow(target_fullsize.permute(1, 2, 0))
        plt.show()
    else:
        f, axarr = plt.subplots(3)
        target_fullsize = t.zeros_like(source)
        target_width = target.size(-1)
        source_width = source.size(-1)
        heatmap_width = heatmap.size(-1)
        heatmap_idx = t.argmax(heatmap)
        target_fullsize_start = int(heatmap_idx * (source_width/heatmap_width) - target_width//2)
        target_fullsize[:, :, max(target_fullsize_start, 0):max(target_fullsize_start, 0)+target_width] = target
        axarr[0].imshow(source.permute(1, 2, 0), aspect="auto")
        axarr[1].imshow(target_fullsize.permute(1, 2, 0), aspect="auto")
        resized_indices = np.arange(0, prediction.size(-1)) * (source_width/heatmap_width)
        pred = np.interp(np.arange(0, source_width), resized_indices, prediction.numpy())
        predicted_max = np.argmax(pred)
        axarr[2].axvline(x=predicted_max, ymin=0, ymax=1, c="r")
        axarr[2].plot(pred)
        axarr[2].set_xlim((0, source_width - 1))
        Path(dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(dir + str(name) + "png")
        plt.close()


def plot_displacement(source, target, prediction, displacement=None, name=0, dir="results/0/"):
    f, axarr = plt.subplots(3)
    heatmap_width = prediction.size(-1)
    source_width = source.size(-1)
    resized_indices = np.arange(0, prediction.size(-1)) * (source_width / heatmap_width)
    pred = np.interp(np.arange(0, source_width), resized_indices, prediction.numpy())
    predicted_max = np.argmax(pred)
    predicted_shift = -(256 - predicted_max)
    target_shifted = t.zeros_like(target)
    if displacement is not None:
        displacement = int(displacement)
    if predicted_shift > 0:
        target_shifted[..., predicted_shift:] = target[..., :-predicted_shift]
    elif predicted_shift < 0:
        target_shifted[..., :predicted_shift] = target[..., -predicted_shift:]
    else:
        target_shifted = target
    axarr[0].imshow(source.permute(1, 2, 0), aspect="auto")
    axarr[1].imshow(target_shifted.permute(1, 2, 0), aspect="auto")
    axarr[2].axvline(x=predicted_max, ymin=0, ymax=1, c="r")
    if displacement is not None:
        axarr[2].axvline(x=displacement + 256, ymin=0, ymax=1, c="b", ls="--")
    axarr[2].plot(pred)
    axarr[2].set_xlim((0, source_width - 1))
    Path(dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(dir + str(name) + ".png")
    plt.close()


def get_shift(img_width, crop_width, histogram, crops_idx):
    img_center = img_width//2
    histnum = histogram.size(0)
    histogram = histogram.cpu()
    hist_size = histogram.size(-1)
    hist_center = hist_size
    final_hist = t.zeros(hist_size * 2)
    bin_size = t.zeros_like(final_hist)
    for idx, crop_idx in enumerate(crops_idx):
        crop_to_img = ((crop_idx + crop_width//2) - img_center)/img_width
        crop_displac_in_hist = int(crop_to_img * hist_size)
        final_hist_start = hist_center//2 + crop_displac_in_hist
        final_hist[final_hist_start:final_hist_start+hist_size] += histogram[histnum - idx - 1]
        bin_size[final_hist_start:final_hist_start+hist_size] += 1
    final_hist /= bin_size
    return final_hist[hist_size//2:-hist_size//2]


def affine(img, rotate, translate):
    # rotate - deg, translate - [width, height]
    device = img.device
    rotated = K.rotate(img, t.tensor(rotate, device=device), align_corners=False)
    return K.translate(rotated, t.tensor([translate], device=device), align_corners=False)
