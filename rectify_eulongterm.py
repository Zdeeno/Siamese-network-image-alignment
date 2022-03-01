import numpy as np
import torchvision
import torch as t
from model import Siamese, get_custom_CNN, load_model
from matplotlib import pyplot as plt
from torchvision.transforms import Resize
from parser_eulongterm import FrameLongterm
from utils import plot_similarity, save_imgs
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from scipy import interpolate


def get_pad(crop):
    return (crop - 8) // 16

# network params
MODEL = "/home/zdeeno/Documents/Work/alignment/results_siam_cnn/model_150_noBN.pt"
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
WIDTH = 640
CROP_SIZE = 632  # [56 + 16*i for i in range(5)]
FRACTION = 8
SHIFT = FRACTION // 2
PAD = get_pad(CROP_SIZE)
OUTPUT_SIZE = 80

# program params
VISUALIZE = False
SAVE_SOURCE = False
START_IDX = 1  # for restarting session
STEP = 1  # frames in one step
DISTANCE = 10  # GPS distance between positions
SEARCH_SIZE = 32
VIDEO1 = "/home/zdeeno/Documents/Datasets/eu_longterm/0719/images"
GPS1 = "/home/zdeeno/Documents/Datasets/eu_longterm/0719/utbm_robocar_dataset_20180719_noimage_gps.csv"
VIDEO2 = "/home/zdeeno/Documents/Datasets/eu_longterm/0502B/images"
GPS2 = "/home/zdeeno/Documents/Datasets/eu_longterm/0502B/utbm_robocar_dataset_20180502_noimage2_gps.csv"
TARGET_PATH = "/home/zdeeno/Documents/Datasets/eu_longterm/"

resize = Resize(480)


def preprocess_for_model(image):
    image = resize(image.to(device)).contiguous().float()
    return image


def get_batches(img1_idx, img2_idx):
    img1_idx = img1_idx
    img1 = video1[img1_idx]
    batch1 = img1.unsqueeze(0)
    batch1 = preprocess_for_model(batch1)
    indices_video2 = [i for i in range(img2_idx - SEARCH_SIZE, img2_idx + SEARCH_SIZE)]
    batch2 = video2[indices_video2]
    batch2 = preprocess_for_model(batch2)
    return batch1, batch2


def time_histogram(batch1, batch2):
    with t.no_grad():
        histograms = model(batch1, batch2[..., SHIFT:SHIFT + CROP_SIZE])
        time_maxs = t.max(histograms, dim=-1)[0]
        # std, mean = t.std_mean(time_maxs, dim=-1, keepdim=True)
        # regularized_time_maxs = (time_maxs - mean) / std
        # time_histogram = t.softmax(regularized_time_maxs, dim=0)
        best_img_idx = t.argmax(time_maxs)
        drift_increment = best_img_idx - SEARCH_SIZE
        final_img1 = batch1[0].cpu()
        final_img2 = batch2[SEARCH_SIZE + drift_increment].cpu()
        return final_img1, final_img2, time_maxs, histograms[best_img_idx]


if __name__ == '__main__':
    # time.sleep(int(3600 * 2.75))

    video1 = FrameLongterm(VIDEO1, GPS1)
    video2 = FrameLongterm(VIDEO2, GPS2, ts_diff=-19000)

    backbone = get_custom_CNN(lp=False, fs=3, ech=256)  # use custom network trained from scratch PAD = 3
    model = Siamese(backbone, padding=PAD, eb=True).to(device)
    model = load_model(model, MODEL).float()
    model.eval()

    step = START_IDX
    gps_idx = 0
    while True:
        out = video1.get_next_position(gps_idx, DISTANCE)
        if out is None:
            break
        else:
            gps_idx, coords, img1_idx = out
            img2_idx, dist = video2.get_nearest_index(*coords)
            print("step", step, "dist", dist, "img1idx", img1_idx, "img2idx", img2_idx, "gpsidx1", gps_idx)
            if not (img2_idx + SEARCH_SIZE > len(video2) or img2_idx - SEARCH_SIZE < 0):
                if dist < 2 * DISTANCE:
                    b1, b2 = get_batches(img1_idx, img2_idx)
                    img1, img2, t_hist, best_hist = time_histogram(b1, b2)

                    # estimate offset
                    f = interpolate.interp1d(np.linspace(0, WIDTH, OUTPUT_SIZE), best_hist.cpu(), kind="cubic")
                    interpolated = f(np.arange(WIDTH))
                    ret = np.argmax(interpolated) - WIDTH // 2

                    # visualize or save
                    if VISUALIZE:
                        plot_similarity(img1, img2, t_hist.cpu(), name=str(step).zfill(6), offset=ret)
                    else:
                        if not SAVE_SOURCE:
                            img1 = None
                        save_imgs(img2, name=str(step).zfill(6), img1=img1, path=TARGET_PATH,
                                  max_val=t.max(t.softmax(t_hist, dim=-1)).cpu().numpy(), offset=ret)
                else:
                    print("Too high distance", dist)
            else:
                print("Search set outside the dataset", img2_idx, len(video2))
        step += 1
