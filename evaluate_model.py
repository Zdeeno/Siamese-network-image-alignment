import torch
import torch as t
from model import Siamese, load_model, get_custom_CNN, jit_load, get_parametrized_model
from torch.utils.data import DataLoader
from parser_grief import ImgPairDataset, CroppedImgPairDataset, ImgPairDatasetOld
from torchvision.transforms import Resize
from tqdm import tqdm
from utils import get_shift, plot_samples, plot_displacement, affine
import numpy as np
from scipy import interpolate
from configs import CONFIG


device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
# device = t.device("cpu")

DATASET = "carlevaris"
VISUALIZE = False
PLOT_IMPORTANCES = False
WIDTH = 512  # - 8
CROP_SIZE = WIDTH - 8
PAD = 31
FRACTION = 8
OUTPUT_SIZE = 64  #  WIDTH // FRACTION
CROPS_MULTIPLIER = 1
BATCHING = CROPS_MULTIPLIER    # this improves evaluation speed by a lot
MASK = t.zeros(OUTPUT_SIZE)
MASK[:PAD] = t.flip(t.arange(0, PAD), dims=[0])
MASK[-PAD:] = t.arange(0, PAD)
MASK = OUTPUT_SIZE - 1 - MASK
MASK = (OUTPUT_SIZE - 1) / MASK.to(device)
LAYER_POOL = False
FILTER_SIZE = 3
EMB_CHANNELS = 256
RESIDUALS = 0

EVAL_LIMIT = 2000
TOLERANCE = 32

MODEL_TYPE = "siam"
MODEL = "model_eunord"
# MODEL = "model_47"


transform = Resize(192)
# transform = Resize(192 * 2)  # 4:3
# transform = Resize((288, 512))
crops_num = int((WIDTH // CROP_SIZE) * CROPS_MULTIPLIER)
crops_idx = np.linspace(0, WIDTH-CROP_SIZE, crops_num, dtype=int) + FRACTION // 2

# crops_idx = np.array([WIDTH // 2 - CROP_SIZE // 2])
# crops_num = 1
print(crops_num, np.array(crops_idx))
histograms = np.zeros((1000, 64))


args = CONFIG("best_params")
NAME = args.nm
LR = 10**-args.lr
BATCH_SIZE = args.bs
NEGATIVE_FRAC = args.nf
# device = args.dev
LAYER_POOL = args.lp
FILTER_SIZE = args.fs
EMB_CHANNELS = args.ech
SMOOTHNESS = args.sm
assert args.res in [0, 1, 2, 3], "Residual type is wrong"
RESIDUALS = args.res


model = get_parametrized_model(LAYER_POOL, FILTER_SIZE, EMB_CHANNELS, RESIDUALS, PAD, device)


def get_histogram(src, tgt):
    target_crops = []
    for crop_idx in crops_idx:
        target_crops.append(tgt[..., crop_idx:crop_idx + CROP_SIZE])
    target_crops = t.cat(target_crops, dim=0)
    batched_source = src.repeat(crops_num // BATCHING, 1, 1, 1)
    # batched_source = t.zeros_like(batched_source)
    # batched_source = src
    histogram = model(batched_source, target_crops, padding=PAD)  # , fourrier=True)
    # histogram = histogram * MASK
    # histogram = t.sigmoid(histogram)
    # std, mean = t.std_mean(histogram, dim=-1, keepdim=True)
    # histogram = (histogram - mean) / std
    histogram = t.softmax(histogram, dim=1)
    return histogram


def get_importance(src, tgt, displac):
    # displac here is in size of embedding (OUTPUT_SIZE)
    histogram = model(src, tgt, padding=PAD, displac=displac).cpu().numpy()
    f = interpolate.interp1d(np.linspace(0, 512, OUTPUT_SIZE), histogram, kind="cubic")
    interpolated = f(np.arange(512))
    return interpolated[0]


def eval_displacement(eval_model=None, path="./results_" + MODEL_TYPE + "/" + MODEL + ".pt", dataset_path=None):
    global model
    # backbone = get_custom_CNN(LAYER_POOL, FILTER_SIZE, EMB_CHANNELS)
    # model = Siamese(backbone, padding=PAD, eb=END_BN).to(device)
    # model = load_model(model, "./results_" + MODEL_TYPE + "/" + MODEL + ".pt")
    if eval_model is not None:
        model = eval_model
    else:
        model = load_model(model, path)

    if dataset_path is not None:
        dataset = ImgPairDataset(path=dataset_path, dataset=DATASET)
    else:
        dataset = ImgPairDataset(dataset=DATASET)

    train_loader = DataLoader(dataset, 1, shuffle=False)

    model.eval()
    with torch.no_grad():
        abs_err = 0
        valid = 0
        idx = 0
        errors = []
        results = []
        for batch in tqdm(train_loader):
            source, target, displ = transform(batch[0].to(device)), transform(batch[1].to(device)), batch[2]

            if abs(displ.numpy()[0]) > 256:
                continue

            # do it in both directions target -> source and source -> target
            histogram = get_histogram(source, target)
            # shift_hist = get_shift(WIDTH, CROP_SIZE, histogram, crops_idx)
            # histograms[idx, :] = histogram.cpu().numpy()
            shift_hist = histogram.cpu()
            # histogram = get_histogram(target, source)
            # shift_hist += t.flip(get_shift(WIDTH, CROP_SIZE, histogram, crops_idx), dims=(-1, ))
            f = interpolate.interp1d(np.linspace(0, 1008, OUTPUT_SIZE), shift_hist, kind="cubic")
            interpolated = f(np.arange(1008))
            # interpolated = np.interp(np.arange(0, 1024), np.linspace(0, 1024, OUTPUT_SIZE), shift_hist.numpy())
            ret = -(np.argmax(interpolated) - 504)
            # print(ret, displ)
            results.append(ret)
            displac_mult = 1024/WIDTH
            tmp_err = (ret - displ.numpy()[0])/displac_mult
            abs_err += abs(tmp_err)
            errors.append(tmp_err)
            if VISUALIZE and abs(ret - displ.numpy()[0]) >= TOLERANCE:
                if PLOT_IMPORTANCES:
                    importances = get_importance(source, target, int(ret/(FRACTION*displac_mult)))
                else:
                    importances = None
                plot_displacement(source.squeeze(0).cpu(),
                                  target.squeeze(0).cpu(),
                                  shift_hist.squeeze(0).cpu(),
                                  displacement=-displ.numpy()[0]/displac_mult,
                                  importance=importances,
                                  name=str(idx),
                                  dir="results_" + MODEL_TYPE + "/eval_" + MODEL + "/")
                print(ret, displ.numpy()[0])
            if abs(ret - displ.numpy()[0]) < TOLERANCE:
                valid += 1
            idx += 1

            if idx > EVAL_LIMIT:
                break

        print("Evaluated:", "\nAbsolute mean error:", abs_err/idx, "\nPredictions in tolerance:", valid*100/idx, "%")
        np.savetxt("results_" + MODEL_TYPE + "/eval_" + MODEL + "/" + DATASET + "_errors.csv", np.array(errors) * 2.0, delimiter=",")
        # np.savetxt("results_" + MODEL_TYPE + "/eval_" + MODEL + "/" + DATASET + "_preds.csv", np.array(errors) * 2.0, delimiter=",")
        # np.savetxt("results_" + MODEL_TYPE + "/eval_" + MODEL + "/" + DATASET + "_histograms.csv", histograms, delimiter=",")
        return abs_err/idx, valid*100/idx


if __name__ == '__main__':
    eval_displacement()
