import torch as t
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.io import read_image, decode_image
import random
from utils import plot_samples
import itertools
from glob import glob
from kornia.augmentation import RandomAffine
from kornia import Hflip
import numpy as np


class ImgPairDataset(Dataset):

    def __init__(self, path="/home/zdeeno/Documents/Datasets/nordland/NORDLAND512/train"):
        super(ImgPairDataset, self).__init__()
        self.width = 512
        self.height = 288

        lvl1_subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
        all_season_images_pths = []
        for subfolder in lvl1_subfolders:
            files = glob(subfolder + '/**/*.png', recursive=True)
            all_season_images_pths.append(files)

        perms = list(itertools.permutations([1, 2, 3]))
        self.data = []
        for pair in perms:
            generated_pairs = [(all_season_images_pths[pair[0]][idx],
                                all_season_images_pths[pair[1]][idx])
                                for idx in range(len(all_season_images_pths[0]))]
            self.data.extend(generated_pairs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if random.random() > 0.5:
            a, b = 0, 1
        else:
            b, a = 0, 1

        source_img = read_image(self.data[idx][a])/255.0
        target_img = read_image(self.data[idx][b])/255.0
        return source_img, target_img


class CroppedImgPairDataset(ImgPairDataset):

    def __init__(self, crop_width, fraction, smoothness, target_pad=False, augment=True, path="/home/zdeeno/Documents/Datasets/nordland/NORDLAND512/train"):
        super(CroppedImgPairDataset, self).__init__(path=path)
        self.crop_width = crop_width
        self.fraction = fraction
        self.smoothness = smoothness
        self.center_mask = 48
        self.use_augment = augment
        self.affine = RandomAffine(t.tensor(10.0), t.tensor([(self.fraction*2)/self.width, (self.fraction*4)/self.width]), align_corners=False)
        self.flip = Hflip()
        self.pad = target_pad

    def __getitem__(self, idx):
        source, target = super(CroppedImgPairDataset, self).__getitem__(idx)

        if self.use_augment:
            source, target = self.augment(source, target)

        cropped_target, crop_start = self.crop_img(target)
        if self.smoothness == 0:
            heatmap = self.get_heatmap(crop_start)
        else:
            heatmap = self.get_smooth_heatmap(crop_start)
        return source, cropped_target, heatmap

    def crop_img(self, img):
        # crop - avoid center (rails) and edges
        crops = [random.randint(self.crop_width, int(self.width/2 - self.center_mask - self.crop_width)),
                 random.randint(int(self.width/2 + self.center_mask), self.width - 2*self.crop_width - 1)]
        crop_start = random.choice(crops)
        return img[:, :, crop_start:crop_start + self.crop_width], crop_start

    def get_heatmap(self, crop_start):
        frac = self.width // self.fraction
        if self.pad:
            frac -= 1
        heatmap = t.zeros(frac).long()
        idx = int((crop_start + self.crop_width//2) / self.fraction)
        heatmap[idx] = 1
        heatmap[idx + 1] = 1
        return heatmap

    def get_smooth_heatmap(self, crop_start):
        surround = self.smoothness * 2
        frac = self.width // self.fraction - 1
        heatmap = t.zeros(frac + surround)
        idx = int((crop_start + self.crop_width//2) / self.fraction) + self.smoothness
        heatmap[idx] = 1
        idxs = np.array([-1, +1])
        for i in range(1, self.smoothness + 1):
            indexes = list(idx + i * idxs)
            for j in indexes:
                if 0 <= j < heatmap.size(0):
                    heatmap[j] = 1 - i * (1/(self.smoothness + 1))
        return heatmap[surround//2:-surround//2]

    def augment(self, source, target):
        if random.random() > 0.5:
            source = self.flip(source)
            target = self.flip(target)
        source = self.affine(source)
        return source.squeeze(0), target


if __name__ == '__main__':
    dataset = CroppedImgPairDataset(64, 16, 3)
    print(len(dataset))
    a, b, heatmap = dataset[1]
    plot_samples(a, b, heatmap, prediction=heatmap)
