import torch as t
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.io import read_image, decode_image
import random


class ImgPairLoader(Dataset):

    def __init__(self, path="/home/zdeeno/Documents/Datasets/grief_jpg"):
        super(ImgPairLoader, self).__init__()
        self.width = 1024

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

        source_img = read_image(self.annotated_img_pairs[idx][a])
        target_img = read_image(self.annotated_img_pairs[idx][b])
        return source_img, target_img, displacement


if __name__ == '__main__':
    dataset = ImgPairLoader()
    import matplotlib.pyplot as plt
    plt.imshow(dataset[0][1].permute(1, 2, 0))
    plt.imshow(dataset[0][0].permute(1, 2, 0))
    plt.show()