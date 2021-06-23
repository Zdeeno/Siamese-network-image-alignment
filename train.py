import torch as t
from data_parser import CroppedImgPairDataset
from model import Siamese
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


BATCH_SIZE = 16
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")


if __name__ == '__main__':
    dataset = CroppedImgPairDataset(32, 8)
    dataloader = DataLoader(dataset, BATCH_SIZE)

    model = Siamese().to(device)
    optimizer = AdamW(model.parameters())
    loss = CrossEntropyLoss()

    def train_loop():
        model.train()
        for batch in tqdm(dataloader):
            source, target, heatmap = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            out = model(source, target)
            l = loss(out, t.argmax(heatmap, dim=-1))
            l.backward()
            optimizer.step()
            optimizer.zero_grad()

    train_loop()