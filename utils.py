import os
from typing import Dict, Optional
import matplotlib.pyplot as plt
import torch
import torch as t
from torch.nn import functional as F
import numpy as np
from pathlib import Path
import kornia as K
import random
from torchvision.utils import save_image
from scipy import interpolate


# class GANAugemntation(t.nn.Module):
#
#     def __init__(self, p=0.5):
#         super(GANAugemntation, self).__init__()
#         self.augmentor = StyleAugmentor()
#         self.p = p
#
#     def forward(self, x):
#         with torch.no_grad():
#             if random.random() < self.p and x.size(-1) > 40:
#                 return self.augmentor(x, alpha=0.25)
#             else:
#                 return x


AUG_P = 0.1

batch_augmentations = t.nn.Sequential(
    K.augmentation.RandomAffine(t.tensor(10.0),
                                t.tensor([0.0, 0.15]),
                                align_corners=False, p=AUG_P),
    K.augmentation.RandomBoxBlur(p=AUG_P),
    K.augmentation.RandomChannelShuffle(p=AUG_P),
    K.augmentation.RandomPerspective(distortion_scale=0.05, p=AUG_P),
    # K.augmentation.RandomPosterize(p=0.2),    CPU only
    K.augmentation.RandomSharpness(p=AUG_P),
    K.augmentation.RandomSolarize(p=AUG_P),
    K.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=AUG_P),
    K.augmentation.RandomGaussianNoise(std=0.1, p=AUG_P),
    K.augmentation.RandomElasticTransform(p=AUG_P),
    # K.augmentation.RandomEqualize(p=0.2),     CPU only
    K.augmentation.RandomGrayscale(p=AUG_P),
    # K.augmentation.RandomErasing(p=AUG_P, scale=(0.1, 0.5))
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
        fx = interpolate.interp1d(np.linspace(0, 512, 64), heatmap, kind="linear")
        heatmap_plot = fx(np.arange(512))
        target_fullsize_start = int(heatmap_idx * (source_width/heatmap_width) - target_width//2)
        target_fullsize[:, :, max(target_fullsize_start, 0):max(target_fullsize_start, 0)+target_width] = target
        axarr[0].imshow(source.permute(1, 2, 0), aspect="auto")
        axarr[1].imshow(target_fullsize.permute(1, 2, 0), aspect="auto")
        resized_indices = np.arange(0, prediction.size(-1)) * (source_width/heatmap_width)
        pred = np.interp(np.arange(0, source_width), resized_indices, prediction.numpy())
        predicted_max = np.argmax(pred)
        # axarr[2].axvline(x=predicted_max, ymin=0, ymax=1, c="r")
        axarr[2].plot(np.arange(-256, 256), pred)
        axarr[2].plot(np.arange(-256, 256), heatmap_plot)
        axarr[2].set_xlim((0, source_width - 1))
        axarr[2].set_xlabel("Displacement [px]")
        axarr[2].set_ylabel("Likelihood [-]")
        axarr[2].set_xlim((-256, 256))
        axarr[2].grid()
        axarr[2].legend(["prediction", "target"])
        f.suptitle("Training example")
        f.tight_layout()
        Path(dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(dir + str(name) + "png")
        plt.close()


def plot_displacement(source, target, prediction, displacement=None, importance=None, name=0, dir="results/0/"):
    if importance is None:
        f, axarr = plt.subplots(3)
    else:
        f, axarr = plt.subplots(4)
    heatmap_width = prediction.size(-1)
    source_width = source.size(-1)
    resized_indices = np.arange(0, prediction.size(-1)) * (source_width / heatmap_width)
    pred = np.interp(np.arange(0, source_width), resized_indices, prediction.numpy())
    predicted_max = np.argmax(pred)
    predicted_shift = -(256 - predicted_max)
    if displacement is not None:
        displacement = int(displacement)

    target_shifted = t.roll(target, predicted_shift, -1)
    axarr[0].imshow(source.permute(1, 2, 0), aspect="auto")
    axarr[1].imshow(target_shifted.permute(1, 2, 0), aspect="auto")
    axarr[2].axvline(x=predicted_max - 256, ymin=0, ymax=1, c="r")
    if displacement is not None:
        axarr[2].axvline(x=displacement, ymin=0, ymax=1, c="b", ls="--")
    axarr[2].plot(np.arange(-256, 256), pred)
    axarr[2].set_xlabel("Displacement [px]")
    axarr[2].set_ylabel("Likelihood [-]")
    axarr[2].grid()
    if displacement is None:
        axarr[2].legend(["prediction", "likelihood"])
    else:
        axarr[2].legend(["prediction", "ground truth", "displacement likelihood"])
    axarr[2].set_xlim((-256, 256))
    if importance is not None:
        axarr[3].plot(importance)
        axarr[3].set_xlim((0, source_width - 1))
    f.suptitle("Evaluation example")
    f.tight_layout()
    Path(dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(dir + str(name) + ".png")
    plt.close()


def plot_similarity(img1, img2, time_histogram, name=None, offset=None):
    f, axarr = plt.subplots(3)
    if offset is not None:
        img2 = t.roll(img2, offset, -1)
    axarr[0].imshow(img1.permute(1, 2, 0), aspect="auto")
    axarr[1].imshow(img2.permute(1, 2, 0), aspect="auto")
    predicted_max = t.argmax(time_histogram)
    max_y = t.max(time_histogram)
    # axarr[2].axvline(x=predicted_max, ymin=0, ymax=max_y, c="r")
    axarr[2].plot(np.arange(-time_histogram.size(0)//2, time_histogram.size(0)//2), time_histogram)
    axarr[2].set_xlabel("offset from $j_k$")
    axarr[2].set_ylabel("similarity")
    axarr[2].grid()
    # axarr[2].set_xlim((0, img1.size(-1) - 1))
    # Path(dir).mkdir(parents=True, exist_ok=True)
    # plt.savefig(dir + str(name) + ".png")
    f.suptitle("Alignment in time")
    f.tight_layout()
    if name is not None:
        plt.savefig("results_aligning/" + name + ".png")
    else:
        plt.show()
    plt.close()


def plot_cuts(img1, img2, suptitle, name=None):
    f, axarr = plt.subplots(2)
    axarr[0].imshow(img1.permute(1, 2, 0), aspect="auto")
    axarr[1].imshow(img2.permute(1, 2, 0), aspect="auto")
    f.suptitle(suptitle)
    if name is not None:
        # plt.savefig("results_cuts/" + name + ".png")
        pass
    else:
        plt.show()
    plt.close()


def save_imgs(img2, name, img1=None, path="/home/zdeeno/Documents/Datasets/eulongterm_rectified", max_val=None, offset=None):
    path1 = os.path.join(path, "0", str(name) + ".png")
    path2 = os.path.join(path, "1", str(name) + ".png")
    if img1 is not None:
        save_image(img1, path1)
    save_image(img2, path2)

    if offset is not None:
        write_str = str(name) + " " + str(max_val) + " " + str(offset) + "\n"
    else:
        write_str = str(name) + " " + str(max_val) + "\n"

    if max_val is not None:
        f = open(os.path.join(path, "quality.txt"), 'a')
        f.write(write_str)


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


def plot_img_pair(img1, img2):
    f, axarr = plt.subplots(2)
    axarr[0].imshow(img1.permute(1, 2, 0), aspect="auto")
    axarr[1].imshow(img2.permute(1, 2, 0), aspect="auto")
    plt.show()

