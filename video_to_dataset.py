import torchvision
import torch as t
from model import Siamese, get_custom_CNN, load_model
from train_siam import get_pad
from matplotlib import pyplot as plt
from torchvision.transforms import Resize
from parser_nordland import VideoDataset, GPSDataset
from utils import plot_similarity

# network params
MODEL = "/home/zdeeno/Documents/Work/alignment/results_siam/model_4.pt"
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
WIDTH = 512
CROP_SIZE = 504  # [56 + 16*i for i in range(5)]
FRACTION = 8
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
STEP = 10  # frames in one step
DISTANCE = 100  # GPS distance between positions
SEARCH_SIZE = 100

# dataset params
FPS = 25
VIDEO1 = "/home/zdeeno/Documents/Datasets/nordland/videos/winter"
START1 = 300  # frame in which train starts
GPS1 = "/home/zdeeno/Documents/Datasets/nordland/gps/winter.csv"
VIDEO2 = "/home/zdeeno/Documents/Datasets/nordland/videos/spring"
START2 = 100
GPS2 = "/home/zdeeno/Documents/Datasets/nordland/gps/spring.csv"
resize = Resize(288)


def preprocess_for_model(image):
    image = image.to(device)
    image = image.permute(0, 3, 1, 2)
    image = resize(image)
    return image


if __name__ == '__main__':

    video1 = VideoDataset(VIDEO1)
    video2 = VideoDataset(VIDEO2)
    gps1 = GPSDataset(GPS1)
    gps2 = GPSDataset(GPS2)
    gps2.set_position(gps1.get_position())

    backbone = get_custom_CNN()  # use custom network trained from scratch PAD = 3
    model = Siamese(backbone, padding=PAD).to(device)
    model = load_model(model, MODEL)

    while True:
        # load close images
        img1_idx, img1_pos = gps1.get_next(DISTANCE, STEP)
        img2_idx, img2_pos = gps2.get_next(DISTANCE, STEP)
        img2_idx = gps2.get_closest_id(img1_pos, img2_idx, set_position=True)
        img1 = video1[START1 + img1_idx * FPS]
        batch2 = video2[START2 + img2_idx * FPS - SEARCH_SIZE:START2 + img2_idx * FPS + SEARCH_SIZE]
        batch1 = img1.repeat(2 * SEARCH_SIZE)

        # get matching images
        histograms = model(batch1, batch2)
        time_histogram = t.softmax(t.max(histograms, dim=-1)[0], dim=0)
        match_increment = t.argmax(time_histogram) - SEARCH_SIZE
        print(match_increment)
        final_img1 = batch1[0]
        final_img2 = batch2[SEARCH_SIZE + match_increment]

        # visualize and save
        plot_similarity(final_img1, final_img2, time_histogram)







