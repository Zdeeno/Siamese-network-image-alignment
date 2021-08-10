import random
import torch as t
from parser_nordland import CroppedImgPairDataset
from model import Siamese, get_custom_CNN, get_pretrained_VGG11, get_super_backbone, save_model, load_model
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from torch.nn.functional import softmax
from tqdm import tqdm
from utils import plot_samples, batch_augmentations


def get_pad(crop):
    return (crop - 8) // 16


BATCH_SIZE = 16
EPOCHS = 1000
LR = 3e-5
EVAL_RATE = 1
CROP_SIZES = [56]  # [56 + 16*i for i in range(5)]
FRACTION = 8
PAD = get_pad(CROP_SIZES[0])
SMOOTHNESS = 3
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
# device = t.device("cpu")
batch_augmentations = batch_augmentations.to(device)
print(CROP_SIZES)

dataset = CroppedImgPairDataset(CROP_SIZES[0], FRACTION, SMOOTHNESS)
val, train = t.utils.data.random_split(dataset, [int(0.1 * len(dataset)), int(0.9 * len(dataset))])
train_loader = DataLoader(train, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val, 1, shuffle=False)

# backbone = get_pretrained_VGG11()   # use pretrained network - PAD = 7
backbone = get_custom_CNN()  # use custom network trained from scratch PAD = 3
model = Siamese(backbone, padding=PAD).to(device)
optimizer = Adam(model.parameters(), lr=LR)
# loss = CrossEntropyLoss()
loss = BCEWithLogitsLoss()
# loss = MSELoss()


def create_true_negatives(target, heatmap, num):
    num = min(num, target.size(0))
    idxs = random.sample([i for i in range(target.size(0))], num)
    num_half = num//2
    tmp = target[idxs[:num_half]]
    target[idxs[:num_half]] = target[idxs[num_half:]]
    target[idxs[num_half:]] = tmp
    heatmap[idxs, :] = 0
    return target, heatmap


def train_loop(epoch):
    global PAD
    model.train()
    loss_sum = 0
    print("Training model epoch", epoch)
    for batch in tqdm(train_loader):
        source, target, heatmap = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        source = batch_augmentations(source)
        # target = batch_augmentations(target)
        out = model(source, target, padding=PAD)
        optimizer.zero_grad()
        heatmap[heatmap > 0] = 1.0
        # print(t.where(heatmap == 1)[1].view(BATCH_SIZE, 3))
        # l = loss(out, t.argmax(heatmap.long(), dim=-1))
        l = loss(out, heatmap)
        loss_sum += l.cpu().detach().numpy()
        l.backward()
        optimizer.step()

        r_choice = random.choice(CROP_SIZES)
        PAD = get_pad(r_choice)
        dataset.set_crop_size(r_choice, smoothness=SMOOTHNESS)

    print("Training of epoch", epoch, "ended with loss", loss_sum / len(train_loader))


def eval_loop(epoch):
    global PAD
    model.eval()
    with t.no_grad():
        print("Evaluating model after epoch", epoch)
        for idx, batch in enumerate(val_loader):
            if idx % 10 == 0:
                source, target, heatmap = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                source = batch_augmentations(source)
                # target = batch_augmentations(target)
                out = model(source, target, padding=PAD)
                out = out.squeeze(0).cpu()
                plot_samples(source.squeeze(0).cpu(),
                             target.squeeze(0).cpu(),
                             heatmap.squeeze(0).cpu(),
                             prediction=out,
                             name=str(idx),
                             dir="results_siam/" + str(epoch) + "/")

                r_choice = random.choice(CROP_SIZES)
                PAD = get_pad(r_choice)
                dataset.set_crop_size(r_choice, smoothness=SMOOTHNESS)

                if idx > 100:
                    break


if __name__ == '__main__':
    LOAD_EPOCH = 0
    # model, optimizer = load_model(model, "/home/zdeeno/Documents/Work/alignment/results_siam/model_" + str(LOAD_EPOCH) + ".pt", optimizer=optimizer)

    for epoch in range(LOAD_EPOCH, EPOCHS):
        save_model(model, "siam", epoch, optimizer)
        if epoch % EVAL_RATE == 0:
            eval_loop(epoch)
        train_loop(epoch)
