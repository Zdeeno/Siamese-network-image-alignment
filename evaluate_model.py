import torch
import torch as t
from model import Siamese, load_model, get_custom_CNN
from torch.utils.data import DataLoader
from parser_grief import ImgPairDataset, CroppedImgPairDataset
from torchvision.transforms import Resize
from tqdm import tqdm
from utils import get_shift, plot_samples, plot_displacement, affine
import numpy as np

WIDTH = 512
CROP_SIZE = 32
PAD = 0
FRACTION = 16
TOLERANCE = 50
MODEL = "model_3"
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
# device = t.device("cpu")

# backbone = get_pretrained_VGG11()   # use pretrained network - PAD = 7
backbone = get_custom_CNN()  # use custom network trained from scratch PAD = 3
model = Siamese(backbone, padding=PAD).to(device)
model = load_model(model, "/home/zdeeno/Documents/Work/alignment/results/" + MODEL + ".pt")

transform = Resize(192)
CROPS_MULTIPLIER = 1
crops_num = int((WIDTH // CROP_SIZE - 2) * CROPS_MULTIPLIER)
crops_idx = [int(i * (CROP_SIZE/CROPS_MULTIPLIER) + CROP_SIZE) for i in range(crops_num)]


def eval_displacement():
    dataset = ImgPairDataset()
    train_loader = DataLoader(dataset, 1, shuffle=False)

    model.eval()
    with torch.no_grad():
        abs_err = 0
        valid = 0
        idx = 0
        errors = []
        for batch in tqdm(train_loader):
            source, target, displ = transform(batch[0].to(device)), transform(batch[1].to(device)), batch[2]

            def get_histogram(src, tgt):
                target_crops = []
                for crop_idx in crops_idx:
                    target_crops.append(tgt[..., crop_idx:crop_idx+CROP_SIZE])
                target_crops = t.cat(target_crops, dim=0)
                batched_source = src.repeat(crops_num, 1, 1, 1)
                histogram = model(batched_source, target_crops)
                histogram = t.sigmoid(histogram)
                histogram[:, 0] = 0 # filter boundary values
                histogram[:, -1] = 0 # filter boundary values
                return histogram

            # do it in both directions target -> source and source -> target
            histogram = get_histogram(source, target)
            shift_hist = get_shift(WIDTH, CROP_SIZE, histogram, crops_idx)
            histogram = get_histogram(target, source)
            shift_hist += t.flip(get_shift(WIDTH, CROP_SIZE, histogram, crops_idx), dims=(-1, ))

            interpolated = np.interp(np.arange(1024), np.linspace(0, 1024, 31), shift_hist)
            ret = -(np.argmax(interpolated) - 512)
            abs_err += abs(ret - displ.numpy()[0])/2
            errors.append((ret - displ.numpy()[0])/2)
            # plot_displacement(source.squeeze(0).cpu(),
            #                   target.squeeze(0).cpu(),
            #                   shift_hist.squeeze(0).cpu(),
            #                   displacement=-displ.numpy()[0]/2,
            #                   name=str(idx),
            #                   dir="results/eval_" + MODEL + "/")
            # print(ret, displ.numpy()[0])

            idx += 1
            if abs(ret - displ.numpy()[0]) < TOLERANCE:
                valid += 1

            if idx > 250:
                np.savetxt("results/eval_" + MODEL + "/errors.csv", errors, delimiter=",")
                break

        print("Evaluated:", "\nAbsolute mean error:", abs_err/idx, "\nPredictions in tolerance:", valid*100/idx, "%")


def eval_heatmap():
    dataset = CroppedImgPairDataset(CROP_SIZE, FRACTION, 0, transforms=transform)
    train_loader = DataLoader(dataset, 1, shuffle=False)

    model.eval()
    with torch.no_grad():
        idx = 0
        for batch in tqdm(train_loader):
            source, target, heatmap = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            histogram = model(source, target)
            histogram = t.sigmoid(histogram)
            histogram[:, 0] = 0  # filter boundary values
            histogram[:, -1] = 0  # filter boundary values

            plot_samples(source.squeeze(0).cpu(),
                         target.squeeze(0).cpu(),
                         heatmap.squeeze(0).cpu(),
                         prediction=histogram.squeeze(0).cpu(),
                         name=str(idx),
                         dir="results/eval_" + MODEL + "/")

            idx += 1


if __name__ == '__main__':
    eval_displacement()