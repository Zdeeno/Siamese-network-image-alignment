from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl
from sklearn.metrics import auc, roc_curve
from matplotlib import image
import matplotlib.patches as patches


mpl.use('Qt5Agg')


DATASET_NAME = "planetarium"
DATASET_PLOT = "planetarium"


def error_distribution(files: [str], names: [str]):
    for file in files:
        array = np.genfromtxt(file, delimiter=',')
        array = np.array(sorted(abs(np.clip(array, -512, 512))))
        print(auc(array, np.linspace(0, 1, len(array))))
        plt.plot(array, np.linspace(0, 1, len(array)) * 100, linewidth=2)

    plt.title("Accuracies on " + DATASET_NAME + " dataset")
    plt.grid()
    plt.xlabel("error [pix]")
    plt.ylabel("inliers [%]")
    # plt.xscale("log")
    plt.xlim([10, 50])
    plt.ylim([70, 100])
    plt.legend(names, loc=4)
    plt.savefig("./accuracy_" + DATASET_NAME + ".eps")
    # plt.show()


def plot_cutout_region(img_path: str, region1: [[int], [int]], region2: [[int], [int]]):
    img = image.imread(img_path)
    f, axarr = plt.subplots(2)
    axarr[0].imshow(img, aspect="auto")
    axarr[0].axvspan(region1[0][0], region1[0][1], color='green', alpha=0.5)
    axarr[0].axvspan(region1[1][0], region1[1][1], color='green', alpha=0.5)
    axarr[0].title.set_text("Cutout restriction for coarse dataset")
    axarr[1].imshow(img, aspect="auto")
    axarr[1].axvspan(region2[0][0], region2[0][1], color='green', alpha=0.5)
    axarr[1].axvspan(region2[1][0], region2[1][1], color='green', alpha=0.5)
    rect = patches.Rectangle((440, 0), 40, 40, linewidth=1, edgecolor='k', facecolor='k')
    axarr[1].add_patch(rect)
    axarr[1].title.set_text("Cutout restriction for rectified dataset")
    f.tight_layout()
    plt.savefig("./cutouts.png")
    plt.close()


if __name__ == '__main__':
    plot_cutout_region("/home/zdeeno/Documents/Datasets/nordland_rectified/spring/008409.png", [[80, 180], [511 - 180, 511 - 80]], [[0, 180], [511 - 180, 511]])
    error_distribution([# "/home/zdeeno/Documents/Work/GRIEF/results/grief_errors.csv",
                        "/home/zdeeno/Documents/Work/GRIEF/results/grief_errors_" + DATASET_PLOT.lower() + ".csv",
                        "/home/zdeeno/Documents/Work/GRIEF/results/sift_errors_" + DATASET_PLOT.lower() + ".csv",
                        # "/home/zdeeno/Documents/Work/alignment/results_siam/eval_model_48/errors.csv",
                        "/home/zdeeno/Documents/Work/alignment/results_siam_cnn/eval_model_47/" + DATASET_PLOT.lower() + "_errors.csv"],
                        # "/home/zdeeno/Documents/Work/SuperPointPretrainedNetwork/superpixel_errors_" + DATASET_PLOT.lower()],
                       # ["grief", "sift", "model_41", "model_47"]
                         ["grief", "sift", "siamese"])
