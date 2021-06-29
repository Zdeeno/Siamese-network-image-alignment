import torch as t
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.io import read_image, decode_image
import random
from utils import plot_samples
import itertools
from glob import glob


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

    def __init__(self, crop_width, fraction, smoothness, path="/home/zdeeno/Documents/Datasets/nordland/NORDLAND512/train"):
        super(CroppedImgPairDataset, self).__init__(path=path)
        self.crop_width = crop_width
        self.fraction = fraction
        self.smoothness = smoothness

    def __getitem__(self, idx):
        source, target = super(CroppedImgPairDataset, self).__getitem__(idx)
        cropped_target, crop_start = self.crop_img(target)
        if self.smoothness == 0:
            heatmap = self.get_heatmap(crop_start)
        else:
            heatmap = self.get_smooth_heatmap(crop_start)
        return source, cropped_target, heatmap

    def crop_img(self, img):
        crop_start = random.randint(self.crop_width, self.width - 2*self.crop_width - 1)
        return img[:, :, crop_start:crop_start + self.crop_width], crop_start

    def get_heatmap(self, crop_start):
        frac = self.width // self.fraction - 1
        heatmap = t.zeros(frac).long()
        idx = int((crop_start + self.crop_width//2) / self.fraction)
        heatmap[idx] = 1
        return heatmap

    def get_smooth_heatmap(self, crop_start):
        surround = self.smoothness * 2
        frac = self.width // self.fraction - 1
        heatmap = t.zeros(frac + surround)
        idx = int((crop_start + self.crop_width//2) / self.fraction) + self.smoothness
        idxs = t.tensor([-1, +1])
        for i in range(self.smoothness):
            for j in idx + i*idxs:
                if 0 <= j < heatmap.size(0):
                    heatmap[j] = 1 - i * (1/self.smoothness)
        return heatmap[surround//2:-surround//2]


if __name__ == '__main__':
    dataset = CroppedImgPairDataset(64, 16, 3)
    print(len(dataset))
    a, b, heatmap = dataset[1]
    plot_samples(a, b, heatmap, prediction=heatmap)
