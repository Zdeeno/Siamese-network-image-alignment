import torchvision
import torch as t
from model import Siamese, get_custom_CNN
from train_siam import get_pad
from matplotlib import pyplot as plt
from torchvision.transforms import Resize
from parser_nordland import VideoDataset


device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
CROP_SIZE = 504  # [56 + 16*i for i in range(5)]
FRACTION = 8
PAD = get_pad(CROP_SIZE)

VIDEO1 = "/home/zdeeno/Documents/Datasets/nordland/videos/winter"
VIDEO2 = "/home/zdeeno/Documents/Datasets/nordland/videos/spring"
resize = Resize(288)


def preprocess_for_model(image):
    image = image.to(device)
    image = image.permute(0, 3, 1, 2)
    image = resize(image)
    return image


if __name__ == '__main__':

    video = torchvision.io.read_video(VIDEO1)[0]
    backbone = get_custom_CNN()  # use custom network trained from scratch PAD = 3
    model = Siamese(backbone, padding=PAD).to(device)
    print(video.size())
    for idx in range(video.size(0)):
        image = video[idx]
        plt.imshow(image)
        plt.show()

