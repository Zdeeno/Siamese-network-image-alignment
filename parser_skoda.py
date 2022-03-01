from torch.utils.data import Dataset
import os
from torchvision.io import read_image
import random
from os import listdir
from os.path import join, isfile


class ImgPairDataset(Dataset):

    def __init__(self, path="/home/zdeeno/Documents/Datasets/skoda",
                 dataset_pair=("2021.08.08.16.52.23", "2021.08.08.17.26.02"), shift=0):
        super(ImgPairDataset, self).__init__()
        self.width = 640
        self.height = 480
        self.shift = shift

        path1 = join(path, dataset_pair[0])
        path2 = join(path, dataset_pair[1])
        self.dataset1 = sorted([join(path1, f) for f in listdir(path1) if isfile(join(path1, f))])
        self.dataset2 = sorted([join(path2, f) for f in listdir(path2) if isfile(join(path2, f))])

        self.dataset_len = min(len(self.dataset1), len(self.dataset2))

    def __len__(self):
        return self.dataset_len - self.shift

    def __getitem__(self, idx):
        source_img = read_image(self.dataset1[idx])/255.0
        target_img = read_image(self.dataset2[idx + self.shift])/255.0
        return source_img, target_img