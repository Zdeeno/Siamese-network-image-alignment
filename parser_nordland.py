import torch as t
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.io import read_image, decode_image
import random
from utils import plot_samples
import itertools
from glob import glob
import kornia as K
import numpy as np
import torchvision
import pandas as pd


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
        self.center_mask = 48
        self.side_mask = 1.2
        self.flip = K.Hflip()

    def __getitem__(self, idx):
        source, target = super(CroppedImgPairDataset, self).__getitem__(idx)
        source, target = self.augment(source, target)

        cropped_target, crop_start = self.crop_img(target)
        if self.smoothness == 0:
            heatmap = self.get_heatmap(crop_start)
        else:
            heatmap = self.get_smooth_heatmap(crop_start)
        return source, cropped_target, heatmap

    def set_crop_size(self, crop_size, smoothness=None):
        self.crop_width = crop_size
        if smoothness is None:
            self.smoothness = (crop_size // 8) - 1
        else:
            self.smoothness = smoothness

    def crop_img(self, img):
        # crop - avoid center (rails) and edges
        crops = [random.randint(int(self.crop_width * self.side_mask), int(self.width / 2 - self.center_mask - self.crop_width)),
                 random.randint(int(self.width / 2 + self.center_mask), int(self.width - (1 + self.side_mask) * self.crop_width - 1))]
        crop_start = random.choice(crops)
        return img[:, :, crop_start:crop_start + self.crop_width], crop_start

    def get_heatmap(self, crop_start):
        frac = self.width // self.fraction
        heatmap = t.zeros(frac).long()
        idx = int((crop_start + self.crop_width//2) * (frac/self.width))
        heatmap[idx] = 1
        heatmap[idx + 1] = 1
        return heatmap

    def get_smooth_heatmap(self, crop_start):
        surround = self.smoothness * 2
        frac = self.width // self.fraction
        heatmap = t.zeros(frac + surround)
        idx = int((crop_start + self.crop_width//2) * (frac/self.width)) + self.smoothness
        heatmap[idx] = 1
        idxs = np.array([-1, +1])
        for i in range(1, self.smoothness + 1):
            indexes = list(idx + i * idxs)
            for j in indexes:
                if 0 <= j < heatmap.size(0):
                    heatmap[j] = 1 - i * (1/(self.smoothness + 1))
        return heatmap[surround//2:-surround//2]

    def augment(self, source, target):
        # crop the logo - this for some reason makes the network diverge on evaluation set
        # source = source[:, 30:, :]
        # target = target[:, 30:, :]
        if random.random() > 0.8:
            target = source.clone()
        if random.random() > 0.5:
            source = self.flip(source)
            target = self.flip(target)
        return source.squeeze(0), target


class VideoDataset(Dataset):

    def __init__(self, folder):
        videos = glob(os.path.join(folder, '*.mp4'))
        self.videos = sorted(videos)
        self.curr_video = torchvision.io.read_video(self.videos[-1])[0]
        self.last_segment_len = self.curr_video.size(0)
        self._get_segment(0)
        self.segment_len = self.curr_video.size(0)
        self.total_len = (len(self.videos) - 1) * self.segment_len + self.last_segment_len
        print("Loaded video dataset with", len(self), "images in", len(self.videos), "videos!")

    def _get_segment(self, idx):
        # idx is index of videofile
        self.curr_video_idx = idx
        print("Processing video file named:", self.videos[self.curr_video_idx])
        self.curr_video = torchvision.io.read_video(self.videos[self.curr_video_idx])[0]

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if type(idx) == list:
            ret = []
            for i in idx:
                ret.append(self.__obtain_index(i))
            return t.stack(ret)
        else:
            return self.__obtain_index(idx)

    def __obtain_index(self, idx):
        desired_seg = idx // self.segment_len
        if desired_seg != self.curr_video_idx:
            self._get_segment(desired_seg)
        seg_idx = idx % self.segment_len
        return self.curr_video[seg_idx]/255.0


class GPSDataset:

    def __init__(self, filename):
        df = pd.read_csv(filename)
        # remove first few data when train not in motion (for easier alignment of gps and video data)
        df = df.drop(df[(df["speed"] == 0) & (df.index < 500)].index)
        self._positions = np.empty((len(df), 2), dtype=int)
        self._positions[:, 0] = df["lat"]
        self._positions[:, 1] = df["lon"]
        self._curr_pos = 0

    def get_next(self, dist, step):
        new_pos = self._curr_pos + step
        while np.linalg.norm(self._positions[self._curr_pos] - self._positions[new_pos]) < dist:
            new_pos += step
        self._curr_pos = new_pos
        return new_pos, self._positions[new_pos]

    def get_closest_id(self, position, hint, set_position=False):
        SPREAD = 10  # in seconds
        if hint - SPREAD < 0:
            SPREAD = hint
        dists = np.linalg.norm(position - self._positions[hint-SPREAD:hint+SPREAD], axis=1)
        min_idx = np.argmin(dists)
        assert not(min_idx == 0 or min_idx == SPREAD * 2 - 1), "Steps are desynchronized, hint is not matching " + str(dists) + " " + str(hint)
        min_idx += (hint - SPREAD)
        if set_position:
            self._curr_pos = min_idx
        return min_idx

    def set_position(self, position):
        dists = np.linalg.norm(position - self._positions)
        self._curr_pos = np.argmin(dists)

    def get_position(self):
        return self._positions[self._curr_pos]


if __name__ == '__main__':
    # dataset = CroppedImgPairDataset(64, 16, 3)
    # print(len(dataset))
    # a, b, heatmap = dataset[1]
    # plot_samples(a, b, heatmap, prediction=heatmap)
    GPSDataset("/home/zdeeno/Documents/Datasets/nordland/videos/gpsData/fall.csv")
