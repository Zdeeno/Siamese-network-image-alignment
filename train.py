import random
import torch as t
from data_parser import CroppedImgPairDataset
from model import Siamese
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.functional import softmax
from tqdm import tqdm
from utils import plot_samples


BATCH_SIZE = 8
EPOCHS = 1000
LR = 1e-4
EVAL_RATE = 5
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
# device = t.device("cpu")

if __name__ == '__main__':
    dataset = CroppedImgPairDataset(64, 8)
    val, train = t.utils.data.random_split(dataset, [int(0.2*len(dataset)), int(0.8*len(dataset)) + 1])
    train_loader = DataLoader(train, BATCH_SIZE)
    val_loader = DataLoader(val, 1)

    model = Siamese().to(device)
    optimizer = AdamW(model.parameters(), lr=LR)
    # loss = CrossEntropyLoss()
    loss = BCEWithLogitsLoss()

    def train_loop(epoch):
        model.train()
        loss_sum = 0
        for batch in tqdm(train_loader):
            source, target, heatmap = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            out = model(source, target)
            # print(out.size(), heatmap.size())
            optimizer.zero_grad()
            # print(t.where(heatmap == 1)[1].view(BATCH_SIZE, 3))
            l = loss(out, heatmap.float())
            loss_sum += l.cpu().detach().numpy()
            l.backward()
            optimizer.step()
        print("Training of epoch", epoch, "ended with loss", loss_sum/len(train_loader))

    def eval_loop(epoch):
        model.eval()
        with t.no_grad():
            for idx, batch in enumerate(tqdm(val_loader)):
                if random.random() < 0.05:
                    try:
                        source, target, heatmap = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                        out = model(source, target)
                        out = softmax(t.sigmoid(out.squeeze(0).cpu()), dim=0)
                        plot_samples(source.squeeze(0).cpu(),
                                     target.squeeze(0).cpu(),
                                     heatmap.squeeze(0).cpu(),
                                     prediction=out,
                                     name=str(idx),
                                     dir="results/" + str(epoch) + "/")
                    except Exception as e:
                        print(e)

    for epoch in range(EPOCHS):
        train_loop(epoch)
        if epoch % EVAL_RATE == 0:
            eval_loop(epoch)
