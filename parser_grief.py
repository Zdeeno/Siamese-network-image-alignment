import torch as t
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.io import read_image, decode_image
import random
from utils import plot_samples


class ImgPairDataset(Dataset):

    def __init__(self, path="/home/zdeeno/Documents/Datasets/grief_jpg"):
        super(ImgPairDataset, self).__init__()
        self.width = 1024
        self.height = 384

        lvl1_subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
        lvl2_subfolders = []
        for subfolder in lvl1_subfolders:
            lvl2_subfolder = [f.path for f in os.scandir(subfolder) if f.is_dir() and "season" in f.name]
            lvl2_subfolders.append(lvl2_subfolder)
        images_displacements = {}
        for subfolder in lvl2_subfolders:
            for subsubfolder in subfolder:
                os.path.join(subsubfolder, "displacements.txt")
                file = open(os.path.join(subsubfolder, "displacements.txt"), 'r')
                displacements = []
                for idx, line in enumerate(file):
                    displacements.append(int(line.split(" ")[0]))
                images_displacements[subsubfolder] = displacements
        season_pairs = []
        for subsubfolder in lvl2_subfolders:
            res = [(a, b) for idx, a in enumerate(subsubfolder) for b in subsubfolder[idx + 1:]]
            season_pairs.extend(res)
        self.annotated_img_pairs = []
        for pair in season_pairs:
            generated_triplets = [(os.path.join(pair[0], str(idx).zfill(9)) + ".jpg",
                                   os.path.join(pair[1], str(idx).zfill(9)) + ".jpg",
                                   images_displacements[pair[1]][idx] - images_displacements[pair[0]][idx])
                                  for idx in range(len(images_displacements[pair[0]]))]
            self.annotated_img_pairs.extend(generated_triplets)

    def __len__(self):
        return len(self.annotated_img_pairs)

    def __getitem__(self, idx):
        if random.random() > 0.5:
            a, b = 0, 1
            displacement = self.annotated_img_pairs[idx][2]
        else:
            b, a = 0, 1
            displacement = -self.annotated_img_pairs[idx][2]

        source_img = read_image(self.annotated_img_pairs[idx][a])/255.0
        target_img = read_image(self.annotated_img_pairs[idx][b])/255.0
        return source_img, target_img, displacement


class CroppedImgPairDataset(ImgPairDataset):

    def __init__(self, crop_width, fraction, smoothness, path="/home/zdeeno/Documents/Datasets/grief_jpg"):
        super(CroppedImgPairDataset, self).__init__(path=path)
        self.crop_width = crop_width
        self.fraction = fraction
        self.smoothness = smoothness

    def __getitem__(self, idx):
        source, target, displac = super(CroppedImgPairDataset, self).__getitem__(idx)
        cropped_target, crop_start = self.crop_img(target)
        if self.smoothness == 0:
            heatmap = self.get_heatmap(crop_start, displac)
        else:
            heatmap = self.get_smooth_heatmap(crop_start, displac)
        return source, cropped_target, heatmap

    def crop_img(self, img):
        crop_start = random.randint(self.crop_width, self.width - 2*self.crop_width - 1)
        return img[:, :, crop_start:crop_start + self.crop_width], crop_start

    def get_heatmap(self, crop_start, displacement):
        frac = self.width // self.fraction - 1
        heatmap = t.zeros(frac).long()
        idx = int((crop_start - displacement + self.crop_width//2) / self.fraction)
        heatmap[idx] = 1
        return heatmap

    def get_smooth_heatmap(self, crop_start, displacement):
        surround = self.smoothness * 2
        frac = self.width // self.fraction - 1
        heatmap = t.zeros(frac + surround)
        idx = int((crop_start - displacement + self.crop_width//2) / self.fraction) + 3
        idxs = t.tensor([-1, +1])
        for i in range(self.smoothness):
            for j in idx + i*idxs:
                if 0 <= j < heatmap.size(0):
                    heatmap[j] = 1 - i * (1/self.smoothness)
        return heatmap[surround//2:-surround//2]


if __name__ == '__main__':
    dataset = CroppedImgPairDataset(64, 8, 3)
    import matplotlib.pyplot as plt
    a, b, c = dataset[0]
    plot_samples(a, b, c)
