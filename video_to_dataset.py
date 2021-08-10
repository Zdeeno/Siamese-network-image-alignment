import numpy as np
import torchvision
import torch as t
from model import Siamese, get_custom_CNN, load_model
from train_siam import get_pad
from matplotlib import pyplot as plt
from torchvision.transforms import Resize
from parser_nordland import VideoDataset, GPSDataset
from utils import plot_similarity
from torch.utils.data import DataLoader, Subset

# network params
MODEL = "/home/zdeeno/Documents/Work/alignment/results_siam/model_5.pt"
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
WIDTH = 512
CROP_SIZE = 504  # [56 + 16*i for i in range(5)]
FRACTION = 8
SHIFT = FRACTION // 2
PAD = get_pad(CROP_SIZE)
OUTPUT_SIZE = WIDTH // FRACTION
CROPS_MULTIPLIER = 1
MASK = t.zeros(OUTPUT_SIZE)
MASK[:PAD] = t.flip(t.arange(0, PAD), dims=[0])
MASK[-PAD:] = t.arange(0, PAD)
MASK = OUTPUT_SIZE - 1 - MASK
MASK = (OUTPUT_SIZE - 1) / MASK.to(device)
print(MASK)

# program params
STEP = 1  # frames in one step
DISTANCE = 25  # GPS distance between positions
SEARCH_SIZE = 50

# dataset params
"""
Frame in video in which the train starts - since gps is trimmed while standing at the begging
SUMMER START -
SPRING START - (2*60 + 52) * FPS_per_GPS
FALL START -
WINTER START - (2*60 + 59) * FPS_per_GPS
"""
FPS_per_GPS = 25
VIDEO1 = "/home/zdeeno/Documents/Datasets/nordland/videos/winter"
START1 = (2*60 + 55) * FPS_per_GPS
GPS1 = "/home/zdeeno/Documents/Datasets/nordland/videos/gpsData/winter.csv"
VIDEO2 = "/home/zdeeno/Documents/Datasets/nordland/videos/spring"
START2 = (2*60 + 59) * FPS_per_GPS
GPS2 = "/home/zdeeno/Documents/Datasets/nordland/videos/gpsData/spring.csv"
resize = Resize(288)


def preprocess_for_model(image):
    image = resize(image.permute(0, 3, 1, 2)).contiguous()
    image = image.to(device).float()
    return image


if __name__ == '__main__':

    video1 = VideoDataset(VIDEO1)
    video2 = VideoDataset(VIDEO2)

    gps1 = GPSDataset(GPS1)
    gps2 = GPSDataset(GPS2)
    gps2.set_position(gps1.get_position())

    backbone = get_custom_CNN()  # use custom network trained from scratch PAD = 3
    model = Siamese(backbone, padding=PAD).to(device)
    model = load_model(model, MODEL).float()
    model.eval()

    step = 0
    while True:
        # load close pairs of images from videos using gps
        img1_idx, img1_pos = gps1.get_next(DISTANCE, STEP)
        img2_idx, img2_pos = gps2.get_next(DISTANCE, STEP)
        img2_idx = gps2.get_closest_id(img1_pos, img2_idx, set_position=True)
        img1_idx = START1 + img1_idx * FPS_per_GPS
        img1 = video1[img1_idx]
        batch1 = img1.repeat(2 * SEARCH_SIZE, 1, 1, 1)
        batch1 = preprocess_for_model(batch1)
        img2_idx = START2 + img2_idx * FPS_per_GPS
        indices_video2 = [i for i in range(img2_idx - SEARCH_SIZE, img2_idx + SEARCH_SIZE)]
        batch2 = video2[indices_video2]
        batch2 = preprocess_for_model(batch2)
        batch2 = batch2[..., SHIFT:SHIFT+CROP_SIZE]

        print("Step", step, " - Comparing images at index", img1_idx, "vs", img2_idx, "with distance:",
              np.linalg.norm(img1_pos - img2_pos), "at pos", img1_pos, "and", img2_pos)

        # get matching images
        with t.no_grad():
            histograms = model(batch1, batch2) * MASK
            time_maxs = t.max(histograms, dim=-1)[0]
            std, mean = t.std_mean(time_maxs, dim=-1, keepdim=True)
            regularized_time_maxs = (time_maxs - mean)/std
            time_histogram = t.softmax(regularized_time_maxs, dim=0)
            match_increment = t.argmax(time_histogram) - SEARCH_SIZE

        # visualize and save
        final_img1 = batch1[0].cpu()
        final_img2 = batch2[SEARCH_SIZE + match_increment].cpu()
        time_histogram = time_histogram.cpu()
        plot_similarity(final_img1, final_img2, time_histogram, name=str(step))
        step += 1
