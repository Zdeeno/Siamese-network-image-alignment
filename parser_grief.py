import torch as t
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.io import read_image, decode_image
import random
from utils import plot_samples
import pandas as pd
import numpy as np
import json
from torchvision.transforms import Resize


class ImgPairDataset(Dataset):

    def __init__(self, path="/home/zdeeno/Documents/Datasets/grief_jpg", dataset=None):
        super(ImgPairDataset, self).__init__()
        self.width = 1024
        self.height = 384

        self.new_annot = get_annotation_dict(os.path.join(path, "annotation.csv"))
        if dataset is None:
            lvl1_subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
        else:
            lvl1_subfolders = [os.path.join(path, dataset)]
        lvl2_subfolders = []
        lvl1_subfolders = sorted(lvl1_subfolders)
        for subfolder in lvl1_subfolders:
            lvl2_subfolder = sorted([f.path for f in os.scandir(subfolder) if f.is_dir() and "season" in f.name])
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
            dataset = pair[0].split("/")[-2]
            season0 = int(pair[0].split("/")[-1].split("_")[-1]) - 1
            season1 = int(pair[1].split("/")[-1].split("_")[-1]) - 1
            if season0 < 0:
                generated_triplets = [(os.path.join(pair[0], str(idx).zfill(9)) + ".jpg",
                                       os.path.join(pair[1], str(idx).zfill(9)) + ".jpg",
                                       -self.new_annot[dataset][season1][idx])
                                      for idx in range(len(images_displacements[pair[0]]))]
            else:
                generated_triplets = [(os.path.join(pair[0], str(idx).zfill(9)) + ".jpg",
                                       os.path.join(pair[1], str(idx).zfill(9)) + ".jpg",
                                       -self.new_annot[dataset][season1][idx] + self.new_annot[dataset][season0][idx])
                                      for idx in range(len(images_displacements[pair[0]]))]
            self.annotated_img_pairs.extend(generated_triplets)

    def __len__(self):
        return len(self.annotated_img_pairs)

    def __getitem__(self, idx):
        if random.random() > 0.0:
            a, b = 0, 1
            displacement = self.annotated_img_pairs[idx][2]
        else:
            b, a = 0, 1
            displacement = -self.annotated_img_pairs[idx][2]

        source_img = read_image(self.annotated_img_pairs[idx][a]) / 255.0
        target_img = read_image(self.annotated_img_pairs[idx][b]) / 255.0
        return source_img, target_img, displacement


class ImgPairDatasetOld(Dataset):
    # mainly for cestlice

    def __init__(self, path="/home/zdeeno/Documents/Datasets/grief_jpg", dataset="cestlice"):
        super().__init__()
        self.width = 1600
        self.height = 1200
        self.resize = Resize(384 * 2)
        self.MAX_IDX = 248

        self.annotated_img_pairs = get_old_annotation(path, dataset)

    def __len__(self):
        return len(self.annotated_img_pairs)

    def __getitem__(self, idx):
        if random.random() > 0.0:
            a, b = 0, 1
            displacement = self.annotated_img_pairs[idx][2]
        else:
            b, a = 0, 1
            displacement = -self.annotated_img_pairs[idx][2]

        source_img = read_image(self.annotated_img_pairs[idx][a]) / 255.0
        target_img = read_image(self.annotated_img_pairs[idx][b]) / 255.0
        return source_img, target_img, -displacement * (1024 / 1600)


class CroppedImgPairDataset(ImgPairDataset):

    def __init__(self, crop_width, fraction, smoothness, path="/home/zdeeno/Documents/Datasets/grief_jpg",
                 transforms=None):
        super(CroppedImgPairDataset, self).__init__(path=path)
        self.width = 512
        self.height = 192
        self.crop_width = crop_width
        self.fraction = fraction
        self.smoothness = smoothness
        self.tr = transforms

    def __getitem__(self, idx):
        source, target, displac = super(CroppedImgPairDataset, self).__getitem__(idx)
        displac = displac / 2
        if self.tr is not None:
            source = self.tr(source)
            target = self.tr(target)
        cropped_target, crop_start = self.crop_img(target)
        if self.smoothness == 0:
            heatmap = self.get_heatmap(crop_start, displac)
        else:
            heatmap = self.get_smooth_heatmap(crop_start, displac)
        return source, cropped_target, heatmap

    def crop_img(self, img):
        crop_start = random.randint(self.crop_width, self.width - 2 * self.crop_width - 1)
        return img[:, :, crop_start:crop_start + self.crop_width], crop_start

    def get_heatmap(self, crop_start, displacement):
        frac = self.width // self.fraction - 1
        heatmap = t.zeros(frac).long()
        idx = int((crop_start - displacement + self.crop_width // 2) / self.fraction)
        if 0 <= idx < 31:
            heatmap[idx] = 1
        return heatmap

    def get_smooth_heatmap(self, crop_start, displacement):
        surround = self.smoothness * 2
        frac = self.width // self.fraction - 1
        heatmap = t.zeros(frac + surround)
        idx = int((crop_start - displacement + self.crop_width // 2) / self.fraction) + 3
        idxs = t.tensor([-1, +1])
        for i in range(self.smoothness):
            for j in idx + i * idxs:
                if 0 <= j < heatmap.size(0):
                    heatmap[j] = 1 - i * (1 / self.smoothness)
        return heatmap[surround // 2:-surround // 2]


def get_old_annotation(path, dataset):
    if dataset is None:
        lvl1_subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    else:
        lvl1_subfolders = [os.path.join(path, dataset)]
    lvl1_subfolders = sorted(lvl1_subfolders)
    lvl2_subfolders = []
    for subfolder in lvl1_subfolders:
        lvl2_subfolder = sorted([f.path for f in os.scandir(subfolder) if f.is_dir() and "season" in f.name])
        lvl2_subfolders.append(lvl2_subfolder)
    images_displacements = {}
    for subfolder in lvl2_subfolders:
        for subsubfolder in subfolder:
            out = np.empty(1000)
            out[:] = np.NaN
            os.path.join(subsubfolder, "displacements.txt")
            file = open(os.path.join(subsubfolder, "displacements.txt"), 'r')
            displacements = []
            for idx, line in enumerate(file):
                displacements.append(int(line.split(" ")[0]))
            # images_displacements[subsubfolder] = displacements
            season_imgs_idxs = np.sort(np.array(
                [int(f.path.split("/")[-1].split(".")[0]) for f in os.scandir(subsubfolder) if
                 f.path.split(".")[-1] == "jpg"]))
            out[season_imgs_idxs] = displacements
            images_displacements[subsubfolder] = list(out)

    season_pairs = []
    for subsubfolder in lvl2_subfolders:
        res = [(a, b) for idx, a in enumerate(subsubfolder) for b in subsubfolder[idx + 1:]]
        season_pairs.extend(res)
    annotated_img_pairs = []
    for pair in season_pairs:
        generated_triplets = [(os.path.join(pair[0], str(idx).zfill(9)) + ".jpg",
                               os.path.join(pair[1], str(idx).zfill(9)) + ".jpg",
                               images_displacements[pair[1]][idx] - images_displacements[pair[0]][idx])
                              for idx in range(len(images_displacements[pair[0]]))]
        for triplet in generated_triplets:
            if os.path.exists(triplet[0]) and os.path.exists(triplet[1]) and triplet[2] != np.NaN:
                annotated_img_pairs.append(triplet)
    return annotated_img_pairs


def get_annotations(old=False, dataset="stromovka"):
    data_len = 330
    data_name = dataset
    dataset = get_old_annotation("/home/zdeeno/Documents/Datasets/grief_jpg", data_name)
    df = pd.read_csv("/home/zdeeno/Documents/Datasets/grief_jpg/annotation.csv")
    new_displac = np.zeros(data_len)
    new_var = np.zeros(data_len)
    img_index = -1
    for entry in df.iterrows():
        json_str = entry[1]["meta_info"].replace("'", "\"")
        entry_dict = json.loads(json_str)
        if entry_dict["season"] == "":
            continue
        if entry_dict["dataset"] == data_name:
            target_season = int(entry_dict["season"][-2:]) - 1
            img_idx = int(entry_dict["place"])
            if img_idx >= data_len:
                continue
            img_index += 1
            diff = 0
            kp_dict1 = json.loads(entry[1]["kp-1"].replace("'", "\""))
            kp_dict2 = json.loads(entry[1]["kp-2"].replace("'", "\""))
            for kp1, kp2 in zip(kp_dict1, kp_dict2):
                diff += (kp1["x"] / 100) * 1024 - (kp2["x"] / 100) * 1024
            mean = diff // len(kp_dict1)
            diff = 0
            for kp1, kp2 in zip(kp_dict1, kp_dict2):
                diff += (((kp1["x"] / 100) * 1024 - (kp2["x"] / 100) * 1024) - mean) ** 2
            var = np.sqrt(diff / len(kp_dict1))
            print(img_index)
            new_displac[img_index] = mean
            new_var[img_index] = var
    if old:
        old_displac = np.zeros(data_len)
        img_idx = -1
        for pair in dataset:
            img_idx += 1
            if img_idx >= data_len:
                continue
            old_displac[img_idx] = pair[2]
        return old_displac, new_displac, new_var
    else:
        return new_displac, new_var


def get_new_old(old=False, dataset="stromovka"):
    data_len = 500
    data_name = dataset
    dataset = get_old_annotation("/home/zdeeno/Documents/Datasets/grief_jpg", data_name)
    new_dict = get_annotation_dict("/home/zdeeno/Documents/Datasets/grief_jpg/annotation.csv")
    # new_dict[data_name].insert(0, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}) #, 6: 0, 7: 0, 8: 0, 9: 0})
    new_displac = np.zeros(data_len)
    old_displac = np.zeros(data_len)
    displac_idx = -1
    for pair in dataset:
        displac_idx += 1
        print(displac_idx)
        ssn1 = int(pair[0].split("/")[-2].split("_")[-1])
        ssn2 = int(pair[1].split("/")[-2].split("_")[-1])
        img_index = int(pair[1].split("/")[-1].split(".")[0])
        if displac_idx >= data_len:
            continue
        old_displac[displac_idx] = pair[2]
        print(img_index)
        print(new_dict)
        new_displac[displac_idx] = new_dict[data_name][ssn1][img_index] # + new_dict[data_name][ssn2][img_index]
    return old_displac, new_displac


def get_annotation_dict(path):
    df = pd.read_csv(path)
    entries = {"stromovka": [{}], "planetarium": [{} for _ in range(11)],
               "carlevaris": [{}], "michigan": [{} for _ in range(11)],
               "cestlice_reduced": [{} for _ in range(9)]}
    for entry in df.iterrows():
        json_str = entry[1]["meta_info"].replace("'", "\"")
        entry_dict = json.loads(json_str)
        dataset_name = entry_dict["dataset"]
        # this is done for first against all annotation
        if "" != entry_dict["season"]:
            target_season = int(entry_dict["season"][-2:]) - 1
        else:
            continue
        img_idx = int(entry_dict["place"])
        diff = 0
        kp_dict1 = json.loads(entry[1]["kp-1"].replace("'", "\""))
        kp_dict2 = json.loads(entry[1]["kp-2"].replace("'", "\""))
        for kp1, kp2 in zip(kp_dict1, kp_dict2):
            diff += (kp1["x"] / 100) * 1024 - (kp2["x"] / 100) * 1024
        mean = diff // len(kp_dict1)
        if dataset_name == "cestlice_reduced":
            entries[dataset_name][target_season][img_idx] = round(mean)
        else:
            entries[dataset_name][target_season][img_idx] = round(mean)
    return entries


if __name__ == '__main__':
    # dataset = CroppedImgPairDataset(129, 8, 3)
    # for i in range(10):
    #     a, b, c = dataset[i]
    #     plot_samples(a, b, c)
    old, new, new_var = get_annotations(old=True, dataset="cestlice_reduced")
    print(np.mean(new_var))
    # entries = get_annotation_dict("/home/zdeeno/Documents/Datasets/grief_jpg/annotation.csv")
    # dataset = ImgPairDataset()
