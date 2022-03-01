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
from utils import plot_img_pair
from torchvision.transforms import Resize


class ImgPairDataset(Dataset):

    def __init__(self, path):
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


class RectifiedImgPairDataset(Dataset):

    def __init__(self, path):
        super(RectifiedImgPairDataset, self).__init__()
        self.width = 512
        self.height = 384
        self.quality_threshold = 0.05
        self.resize = Resize(384)

        valid_subfolders = ["20180502A", "20180502B", "20180713", "20180716", "20180718", "20180719", "20180720", "20190131", "20190418"]
        lvl1_subfolders = [f.path for f in os.scandir(path) if (f.is_dir() and str(f.path).split("/")[-1] in valid_subfolders)]
        all_season_images_pths = {}
        quality_files = {}
        for subfolder in lvl1_subfolders:
            files = glob(subfolder + '/**/quality.csv', recursive=True)
            quality_files[subfolder] = files
        for subfolder in lvl1_subfolders:
            files = glob(subfolder + '/**/*.png', recursive=True)
            all_season_images_pths[subfolder] = files

        # unroll the qualities
        quality_values = {}
        for key in quality_files.keys():
            print("Getting quality of", key)
            tmp = {}
            for row in open(str(key) + "/quality.txt", "r"):
                split_strings = row.split(" ")
                name = str(split_strings[0])
                value = float(split_strings[1])
                displacement = int(split_strings[2][:-1])    # remove end of the line
                tmp[name] = (value, displacement)
            quality_values[key] = tmp

        perms = []
        stuff = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        for L in range(0, len(stuff) + 1):
            for subset in itertools.combinations(stuff, L):
                if len(subset) == 2:
                    perms.append(subset)
        print(perms)
        self.data = []
        self.displacements = []
        unavailable_pair = 0
        for pair in perms:
            subfolder1 = lvl1_subfolders[pair[0]]
            subfolder2 = lvl1_subfolders[pair[1]]
            for filepath in all_season_images_pths[subfolder1]:
                fileindex = filepath.split("/")[-1][:-4]
                try:
                    if quality_values[subfolder1][fileindex][0] > self.quality_threshold and \
                        quality_values[subfolder2][fileindex][0] > self.quality_threshold:
                        pair = (os.path.join(subfolder1, fileindex + ".png"),
                                os.path.join(subfolder2, fileindex + ".png"))
                        self.data.append(pair)
                        self.displacements.append(quality_values[subfolder1][fileindex][1] - quality_values[subfolder2][fileindex][1])
                    else:
                        unavailable_pair += 1
                except KeyError as err:
                    # print(err, subfolder1, fileindex, subfolder2, fileindex)
                    unavailable_pair += 1
        print("Dataset has", unavailable_pair, "unavailable combinations")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if random.random() > 0.5:
            a, b = 0, 1
        else:
            b, a = 0, 1

        source_img = self.resize(read_image(self.data[idx][a])/255.0)
        target_img = self.resize(read_image(self.data[idx][b])/255.0)

        # crop the hood
        source_img = source_img[:, :-96, :]
        target_img = target_img[:, :-96, :]
        return source_img, target_img, self.displacements[idx]


class RectifiedEULongterm(RectifiedImgPairDataset):

    def __init__(self, crop_width, fraction, smoothness, path=None):
        super(RectifiedEULongterm, self).__init__(path)
        self.crop_width = crop_width
        self.fraction = fraction
        self.smoothness = smoothness
        self.center_mask = 48
        self.flip = K.Hflip()

    def __getitem__(self, idx):
        source, target, displacement = super(RectifiedEULongterm, self).__getitem__(idx)
        # flipping with displacement?
        # source, target = self.augment(source, target)

        cropped_target, crop_start = self.crop_img(target, displacement)
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

    def crop_img(self, img, displac):
        # crop - avoid asking for unavailable crop
        if displac >= 0:
            crops = [random.randint(0, int(self.width - self.crop_width - 1) - displac)]
        else:
            crops = [random.randint(0 - displac, int(self.width - self.crop_width - 1))]

        crop_start = random.choice(crops)
        crop_out = crop_start + displac
        # crop_start = random.randint(0, self.width - self.crop_width - 1)
        return img[:, :, crop_start:crop_start + self.crop_width], crop_out

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
        source[:, :32, -64:] = (t.randn((3, 32, 64)) / 4 + 0.5).clip(0.2, 0.8)
        if random.random() > 0.95:
            target = source.clone()
        if random.random() > 0.5:
            source = self.flip(source)
            target = self.flip(target)
        return source.squeeze(0), target


class FrameLongterm(Dataset):
    def __init__(self, images_path, gps_csv, ts_diff=None):
        self.__ts_resize = 6  # from ns to ms
        self.imgs_path = images_path
        images = glob(os.path.join(images_path, '*.jpg'))
        self.images = sorted(images)
        self.img_name = [os.path.split(pth)[1].split(".")[0] for pth in self.images]
        self.img_ts = t.tensor([int(os.path.split(pth)[1].split(".")[0][:-self.__ts_resize]) for pth in self.images]).long()
        self.gps_df = pd.read_csv(gps_csv)
        print("Loaded images dataset with", len(self), "images!")
        self.lats = t.tensor(self.gps_df["latitude"].to_numpy())
        self.lons = t.tensor(self.gps_df["longitude"].to_numpy())
        if ts_diff is None:
            self.ts_diff = 0
        else:
            self.ts_diff = t.min(self.img_ts) - (self.gps_df["timestamp"][0] * 10**-self.__ts_resize) + ts_diff
        print(self.ts_diff)

    def __len__(self):
        return len(self.img_ts)

    def __getitem__(self, idx):
        if type(idx) == list:
            ret = []
            for i in idx:
                ret.append(read_image(os.path.join(self.imgs_path, str(self.img_name[i]) + ".jpg"))/255.0)
            return t.stack(ret)
        else:
            return read_image(os.path.join(self.imgs_path, str(self.img_name[idx]) + ".jpg"))/255.0

    @staticmethod
    def distance(lat1, lon1, lat2, lon2):
        def deg_to_rad(deg):
            return (deg / 360) * 2 * np.pi
        def rad_to_deg(rad):
            return (rad * 360) / (2 * np.pi)
        theta = lon1 - lon2
        dist = t.sin(deg_to_rad(lat1)) * t.sin(deg_to_rad(lat2)) + t.cos(
            deg_to_rad(lat1)) * t.cos(deg_to_rad(lat2)) * t.cos(deg_to_rad(theta))
        dist[dist > 1] = 0.0
        dist = t.acos(dist)
        dist = rad_to_deg(dist)
        return dist * 60 * 1.1515 * 1.609344 * 1000

    def get_nearest_index(self, lat, lon):
        distances = self.distance(lat, lon, self.lats, self.lons)
        min_gps_idx = t.argmin(distances)
        gps_timestamp = self.gps_df["timestamp"][min_gps_idx.numpy()] / 10**self.__ts_resize
        ts_diff = abs(t.tensor(gps_timestamp).long() - self.img_ts + self.ts_diff)
        closes_idx = t.argmin(ts_diff)
        dist = self.distance(lat, lon, self.lats[min_gps_idx], self.lons[min_gps_idx])
        return closes_idx, dist

    def get_next_position(self, gps_idx, distance):
        lat, lon = self.lats[gps_idx], self.lons[gps_idx]
        idx = 1
        if gps_idx + idx >= self.lats.size()[0]:
            return None
        new_lat, new_lon = self.lats[gps_idx + idx], self.lons[gps_idx + idx]
        dist = self.distance(lat, lon, new_lat, new_lon)
        while dist < distance:
            idx += 1
            if gps_idx + idx >= self.lats.size()[0]:
                return None
            new_lat, new_lon = self.lats[gps_idx + idx], self.lons[gps_idx + idx]
            dist = self.distance(lat, lon, new_lat, new_lon)
        gps_timestamp = self.gps_df["timestamp"][gps_idx + idx] / 10**self.__ts_resize
        ts_diff = abs(t.tensor(gps_timestamp).long() - self.img_ts + self.ts_diff)
        closes_idx = t.argmin(ts_diff)
        return gps_idx + idx, (new_lat, new_lon), closes_idx


if __name__ == '__main__':
    # dataset = CroppedImgPairDataset(64, 16, 3)
    # print(len(dataset))
    # a, b, heatmap = dataset[1]
    # plot_samples(a, b, heatmap, prediction=heatmap)
    data = RectifiedImgPairDataset()
    print(len(data))
    plot_img_pair(*data[100])
