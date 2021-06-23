import matplotlib.pyplot as plt
import torch as t
from torch.nn import functional as F
import numpy as np
from pathlib import Path


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
        target_fullsize[:, :, max(target_fullsize_start, 0):target_fullsize_start+target_width] = target
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