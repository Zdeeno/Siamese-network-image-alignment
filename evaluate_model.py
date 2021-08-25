import torch
import torch as t
from model import Siamese, load_model, get_custom_CNN, get_super_backbone, TeacherStudent
from torch.utils.data import DataLoader
from parser_grief import ImgPairDataset, CroppedImgPairDataset
from torchvision.transforms import Resize
from tqdm import tqdm
from utils import get_shift, plot_samples, plot_displacement, affine
import numpy as np
from scipy import interpolate

"""
Right now the best performance achieves model_20_top with 88 crop size, but the best is to train new model and set
crop size to 504
"""

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
# device = t.device("cpu")

VISUALIZE = True
WIDTH = 512
CROP_SIZE = WIDTH # WIDTH - 8
PAD = (CROP_SIZE - 8) // 16
FRACTION = 8
OUTPUT_SIZE = 64  #  WIDTH // FRACTION
CROPS_MULTIPLIER = 1
BATCHING = CROPS_MULTIPLIER    # this improves evaluation speed by a lot
MASK = t.zeros(OUTPUT_SIZE)
MASK[:PAD] = t.flip(t.arange(0, PAD), dims=[0])
MASK[-PAD:] = t.arange(0, PAD)
MASK = OUTPUT_SIZE - 1 - MASK
MASK = (OUTPUT_SIZE - 1) / MASK.to(device)
print(MASK)

EVAL_LIMIT = 1000
TOLERANCE = 50

MODEL_TYPE = "siam"
MODEL = "model_10"

# backbone = get_pretrained_VGG11()   # use pretrained network - PAD = 7
backbone = get_custom_CNN()  # use custom network trained from scratch PAD = 3
# backbone = get_super_backbone()
if MODEL_TYPE == "siam":
    model = Siamese(backbone, padding=PAD).to(device)
model = load_model(model, "/home/zdeeno/Documents/Work/alignment/results_" + MODEL_TYPE + "/" + MODEL + ".pt")

transform = Resize(192)
# transform = Resize(192 * 2)
# transform = Resize((288, 512))
crops_num = int((WIDTH // CROP_SIZE) * CROPS_MULTIPLIER)
crops_idx = np.linspace(0, WIDTH-CROP_SIZE, crops_num, dtype=int)#  + FRACTION // 2

# crops_idx = np.array([WIDTH // 2 - CROP_SIZE // 2])
# crops_num = 1
print(crops_num, np.array(crops_idx))


def eval_displacement():
    dataset = ImgPairDataset(dataset="stromovka")
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
                batched_source = src.repeat(crops_num//BATCHING, 1, 1, 1)
                # batched_source = t.zeros_like(batched_source)
                # batched_source = src
                histogram = model(batched_source, target_crops, fourrier=True)
                # histogram = histogram * MASK
                # histogram = t.sigmoid(histogram)
                # std, mean = t.std_mean(histogram, dim=-1, keepdim=True)
                # histogram = (histogram - mean)/std
                return histogram

            # do it in both directions target -> source and source -> target
            histogram = get_histogram(source, target)
            # shift_hist = get_shift(WIDTH, CROP_SIZE, histogram, crops_idx)
            shift_hist = histogram.cpu()
            # histogram = get_histogram(target, source)
            # shift_hist += t.flip(get_shift(WIDTH, CROP_SIZE, histogram, crops_idx), dims=(-1, ))

            f = interpolate.interp1d(np.linspace(0, 1024, OUTPUT_SIZE), shift_hist, kind="cubic")
            interpolated = f(np.arange(1024))
            # interpolated = np.interp(np.arange(0, 1024), np.linspace(0, 1024, OUTPUT_SIZE), shift_hist.numpy())
            ret = -(np.argmax(interpolated) - 512)
            displac_mult = 1024/WIDTH
            tmp_err = (ret - displ.numpy()[0])/displac_mult
            abs_err += abs(tmp_err)
            errors.append(tmp_err)
            if VISUALIZE and abs(ret - displ.numpy()[0]) >= TOLERANCE:
                plot_displacement(source.squeeze(0).cpu(),
                                  target.squeeze(0).cpu(),
                                  shift_hist.squeeze(0).cpu(),
                                  displacement=-displ.numpy()[0]/displac_mult,
                                  name=str(idx),
                                  dir="results_" + MODEL_TYPE + "/eval_" + MODEL + "/")
                print(ret, displ.numpy()[0])

            idx += 1
            if abs(ret - displ.numpy()[0]) < TOLERANCE:
                valid += 1

            if idx > EVAL_LIMIT:
                break

        print("Evaluated:", "\nAbsolute mean error:", abs_err/idx, "\nPredictions in tolerance:", valid*100/idx, "%")
        np.savetxt("results_" + MODEL_TYPE + "/eval_" + MODEL + "/errors.csv", np.array(errors) * 2.0, delimiter=",")


if __name__ == '__main__':
    eval_displacement()