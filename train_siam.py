#!/usr/bin/env python3.9
import copy

import numpy as np
import torch as t
from parser_nordland import RectifiedNordland
from parser_compound import CompoundDataset
from parser_eulongterm import RectifiedEULongterm
from model import Siamese, get_custom_CNN, save_model, load_model, jit_save, jit_load, get_parametrized_model
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from torch.nn.functional import softmax
from tqdm import tqdm
from utils import plot_samples, batch_augmentations
import argparse
from evaluate_model import eval_displacement
import wandb
import time


def get_pad(crop):
    return (crop - 8) // 16


VISUALISE = True
WANDB = False
NAME = "siam"
BATCH_SIZE = 32  # higher better
EPOCHS = 128
LR = 4.5
EVAL_RATE = 1
CROP_SIZES = [40]  # [56 + 16*i for i in range(5)]
FRACTION = 8
PAD = get_pad(CROP_SIZES[0])
SMOOTHNESS = 2
NEGATIVE_FRAC = 1/3
LAYER_POOL = True
FILTER_SIZE = 3
EMB_CHANNELS = 128
RESIDUAL = 2

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
# device = t.device("cpu")
batch_augmentations = batch_augmentations.to(device)
print(CROP_SIZES)


parser = argparse.ArgumentParser(description='Training script for image alignment.')
parser.add_argument('--nm', type=str, help="name of the model", default=NAME)
parser.add_argument('--lr', type=float, help="learning rate", default=LR)
parser.add_argument('--bs', type=int, help="batch size", default=BATCH_SIZE)
parser.add_argument('--nf', type=float, help="negative fraction", default=NEGATIVE_FRAC)
# parser.add_argument('--dev', type=str, help="device", required=True)
parser.add_argument('--lp', type=bool, help="layer pooling", default=LAYER_POOL)
parser.add_argument('--fs', type=int, help="filter size", default=FILTER_SIZE)
parser.add_argument('--ech', type=int, help="embedding channels", default=EMB_CHANNELS)
parser.add_argument('--sm', type=int, help="smoothness of target", default=SMOOTHNESS)
parser.add_argument('--cs', type=int, help="crop size", default=CROP_SIZES[0])
parser.add_argument('--res', type=int, help="0 - no residual, 1 - residual 2 - S&E layer, 3 - both", default=RESIDUAL)


args = parser.parse_args()

print("Argument values: \n", args)
NAME = args.nm
LR = 10**-args.lr
BATCH_SIZE = args.bs
NEGATIVE_FRAC = args.nf
# device = args.dev
LAYER_POOL = args.lp
FILTER_SIZE = args.fs
EMB_CHANNELS = args.ech
CROP_SIZES[0] = args.cs
SMOOTHNESS = args.sm
assert args.res in [0, 1, 2, 3], "Residual type is wrong"
RESIDUAL = args.res
EVAL_PATH = "/home/zdeeno/Documents/Datasets/grief_jpg"


dataset = RectifiedNordland(CROP_SIZES[0], FRACTION, SMOOTHNESS,
                            path="/home/zdeeno/Documents/Datasets/nordland_rectified")
# dataset = CompoundDataset(CROP_SIZES[0], FRACTION, SMOOTHNESS,
#                           nordland_path="/home/zdeeno/Documents/Datasets/nordland_rectified",
#                           eu_path="/home/zdeeno/Documents/Datasets/eulongterm_rectified")
# dataset = RectifiedEULongterm(CROP_SIZES[0], FRACTION, SMOOTHNESS,
#                               path="/home/zdeeno/Documents/Datasets/eulongterm_rectified")
val, train = t.utils.data.random_split(dataset, [int(0.05 * len(dataset)), int(0.95 * len(dataset)) + 1])
train_loader = DataLoader(train, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val, BATCH_SIZE, shuffle=False)

# backbone = get_pretrained_VGG11()   # use pretrained network - PAD = 7
# backbone = get_custom_CNN(LAYER_POOL, FILTER_SIZE, EMB_CHANNELS, LAYER_NORM, RESIDUAL)
# backbone = get_UNet()
# model = Siamese(backbone, padding=PAD, eb=END_BN).to(device)
model = get_parametrized_model(LAYER_POOL, FILTER_SIZE, EMB_CHANNELS, RESIDUAL, PAD, device)
optimizer = AdamW(model.parameters(), lr=LR)
# loss = CrossEntropyLoss()
loss = BCEWithLogitsLoss()
in_example = (t.zeros((1, 3, 512, 512)).to(device).float(), t.zeros((1, 3, 512, 512)).to(device).float())


def hard_negatives(batch, heatmaps):
    if batch.shape[0] == BATCH_SIZE - 1:
        num = int(BATCH_SIZE * NEGATIVE_FRAC)
        if num % 2 == 1:
            num -= 1
        indices = t.tensor(np.random.choice(np.arange(0, BATCH_SIZE), num), device=device)
        heatmaps[indices, :] = 0.0
        tmp_t = t.clone(batch[indices[:num//2]])
        batch[indices[:num//2]] = batch[indices[num//2:]]
        batch[indices[num//2:]] = tmp_t
        return batch, heatmaps
    else:
        return batch, heatmaps


def train_loop(epoch):
    PAD = get_pad(CROP_SIZES[0])
    model.train()
    loss_sum = 0
    print("Training model epoch", epoch)
    generation = 0
    for batch in tqdm(train_loader):
        source, target, heatmap = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        source = batch_augmentations(source)
        if NEGATIVE_FRAC > 0.01:
            batch, heatmap = hard_negatives(source, heatmap)
        # target = batch_augmentations(target)
        out = model(source, target, padding=PAD)
        optimizer.zero_grad()
        # heatmap[heatmap > 0] = 1.0
        # print(t.where(heatmap == 1)[1].view(BATCH_SIZE, 3))
        # l = loss(out, t.argmax(heatmap.long(), dim=-1))
        l = loss(out, heatmap)
        loss_sum += l.cpu().detach().numpy()
        l.backward()
        optimizer.step()

        # r_choice = random.choice(CROP_SIZES)
        # PAD = get_pad(r_choice)
        # dataset.set_crop_size(r_choice, smoothness=SMOOTHNESS)
        # generation += 1
        # if generation > 2:
        #     break

    if epoch % EVAL_RATE == 0 and WANDB:
        wandb.log({"epoch": epoch, "train_loss": loss_sum / len(train_loader)})
    print("Training of epoch", epoch, "ended with loss", loss_sum / len(train_loader))


def eval_loop(epoch):
    PAD = get_pad(CROP_SIZES[0])
    model.eval()
    with t.no_grad():
        print("Validating model after epoch", epoch)
        loss_sum = 0
        idx = 0
        for batch in tqdm(val_loader):
            source, target, heatmap = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            source = batch_augmentations(source)
            # target = batch_augmentations(target)
            # batch, heatmap = hard_negatives(source, heatmap)
            out = model(source, target, padding=PAD)
            # out = t.sigmoid(out.squeeze(0).cpu())
            l = loss(out, heatmap)
            loss_sum += l.cpu().detach().numpy()
            idx += 1

            if VISUALISE:
                plot_samples(source[0].cpu(),
                             target[0].cpu(),
                             heatmap[0].cpu(),
                             prediction=out[0].cpu(),
                             name=str(idx),
                             dir="results_" + NAME + "/" + str(epoch) + "/")
            #
            # r_choice = random.choice(CROP_SIZES)
            # PAD = get_pad(r_choice)
            # dataset.set_crop_size(r_choice, smoothness=SMOOTHNESS)
            # break

        print("Evaluating on test set")
        mae, acc = eval_displacement(eval_model=model, dataset_path=EVAL_PATH)
        if WANDB:
            wandb.log({"val_loss": loss_sum/len(val_loader), "MAE": mae, "Accuracy": acc})
            wandb.watch(model)
        print("Epoch", epoch, "validation loss", loss_sum/len(val_loader), "MAE", mae, "Accuracy", acc)

    return mae


LOAD_EPOCH = 0
model, optimizer = load_model(model, "/home/zdeeno/Documents/Work/alignment/results_siam/model_150_noBN.pt", optimizer=optimizer)

if WANDB:
    wandb.init(project="alignment", entity="zdeeno", config=vars(args))

lowest_err = 0
best_model = None

for epoch in range(LOAD_EPOCH, EPOCHS):
    if epoch % EVAL_RATE == 0:
        err = eval_loop(epoch)
        if err < lowest_err:
            lowest_err = err
            best_model = copy.deepcopy(model)
            save_model(model, NAME, err, optimizer)

    train_loop(epoch)
