import numpy as np
import torchvision
import torch as t
from model import Siamese, get_custom_CNN, load_model
from matplotlib import pyplot as plt
from torchvision.transforms import Resize
from parser_nordland import VideoDataset, FrameNordland
from utils import plot_similarity, save_imgs
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

def get_pad(crop):
    return (crop - 8) // 16

# network params
MODEL = "/home/zdeeno/Documents/Work/alignment/results_siam_cnn/model_47.pt"
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
VISUALIZE = True
START_IDX = 10000  # for restarting session
STEP = 1  # frames in one step
DISTANCE = 25  # GPS distance between positions
SEARCH_SIZE = 8
OUTPUT_THRESHOLD = 135
SHIFT_THRESHOLD = 60

# dataset params
"""
Frame in video in which the train starts moving - since gps is trimmed while standing at the begging
SUMMER VIDEO:
SPRING VIDEO: (2*60 + 55) * FPS_per_GPS ... ((9 * 3600) + (53 * 60) + 24) * FPS_per_GPS
FALL VIDEO:
WINTER VIDEO: (2*60 + 57) * FPS_per_GPS ... ((9 * 3600) + (53 * 60) + 26) * FPS_per_GPS
"""
FPS_per_GPS = 25
START = (2*60 + 55) * FPS_per_GPS
VIDEO1 = "/home/zdeeno/Documents/Datasets/nordland/videos/spring"
GPS1 = "/home/zdeeno/Documents/Datasets/nordland/videos/gpsData/spring.csv"
VIDEO2 = "/home/zdeeno/Documents/Datasets/nordland/videos/winter"
GPS2 = "/home/zdeeno/Documents/Datasets/nordland/videos/gpsData/winter.csv"
resize = Resize(288)


# TODO: fix CNN init and image saving


def preprocess_for_model(image):
    image = resize(image.to(device)).contiguous().float()
    return image


def get_batches(img1_idx, shift):
    img1_idx = img1_idx
    img1, name_id = video1[img1_idx]
    batch1 = img1.unsqueeze(0)
    batch1 = preprocess_for_model(batch1)
    img2_idx = video2.get_nearest_idx(name_id)
    indices_video2 = [i for i in range(img2_idx - SEARCH_SIZE + shift, img2_idx + SEARCH_SIZE + shift)]
    batch2 = video2[indices_video2]
    batch2 = preprocess_for_model(batch2)
    return batch1, batch2


def time_histogram(batch1, batch2):
    with t.no_grad():
        histograms = model(batch1, batch2[..., SHIFT:SHIFT + CROP_SIZE]) * MASK
        time_maxs = t.max(histograms, dim=-1)[0]
        # std, mean = t.std_mean(time_maxs, dim=-1, keepdim=True)
        # regularized_time_maxs = (time_maxs - mean) / std
        # time_histogram = t.softmax(regularized_time_maxs, dim=0)
        drift_increment = t.argmax(time_maxs) - SEARCH_SIZE
        final_img1 = batch1[0].cpu()
        final_img2 = batch2[SEARCH_SIZE + drift_increment].cpu()
        return final_img1, final_img2, time_maxs


if __name__ == '__main__':
    import time
    # time.sleep(int(3600 * 2.75))

    video1 = FrameNordland(VIDEO1)
    video2 = FrameNordland(VIDEO2)

    backbone = get_custom_CNN()  # use custom network trained from scratch PAD = 3
    model = Siamese(backbone, padding=PAD).to(device)
    model = load_model(model, MODEL).float()
    model.eval()

    step = START_IDX
    img1_idx = 0
    shift = t.tensor(0, device=device)  # between videos -34
    print(len(video1))
    for i in tqdm(list(np.arange(START, len(video1), SEARCH_SIZE - 1, dtype=int)[START_IDX:])):
        img1_idx = i

        print("Step", step, " - Comparing images at index", img1_idx, "curr shift:", shift.cpu().numpy())

        # get best matching images according to current drift
        batch1, batch2 = get_batches(img1_idx, shift)
        final_img1, final_img2, t_hist = time_histogram(batch1, batch2)
        if abs(shift) > SHIFT_THRESHOLD and t.max(t_hist) < OUTPUT_THRESHOLD:
            batch1, batch2 = get_batches(img1_idx, 0)
            suggest_img1, suggest_img2, suggest_hist = time_histogram(batch1, batch2)
            if t.max(suggest_hist) > t.max(t_hist):
                print("Resetting the shift!")
                final_img1, final_img2, t_hist, shift = suggest_img1, suggest_img2, suggest_hist, 0

        shift_increment = t.argmax(t_hist) - SEARCH_SIZE
        shift += shift_increment//3

        # visualize and save
        if VISUALIZE:
            plot_similarity(final_img1, final_img2, t_hist.cpu(), name=str(step).zfill(6))
        else:
            save_imgs(final_img1, final_img2, name=str(step).zfill(6), max_val=t.max(t.softmax(t_hist, dim=-1)).cpu().numpy())
        step += 1
