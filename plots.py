from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl
from sklearn.metrics import auc, roc_curve
from matplotlib import image
import matplotlib.patches as patches
import pandas as pd
import os
import json

# mpl.use('Qt5Agg')


DATASET_NAME = "carlevaris"
DATASET_PLOT = "carlevaris"
ERROR_CAP = 512
from parser_grief import get_new_old


def error_distribution(files: [str], names: [str]):
    old, new = get_new_old(dataset=DATASET_NAME)
    print(old, new)
    diff = old + new
    print(diff.shape)

    for file in files:
        if len(file) > 1 and file[1]:
            array = np.genfromtxt(file[0], delimiter=',')[:500] - diff
        else:
            array = np.genfromtxt(file[0], delimiter=',')
        array = np.array(sorted(abs(np.clip(array, -ERROR_CAP, ERROR_CAP))))
        print(auc(array, np.linspace(0, 1, len(array))))
        out_fracs = np.zeros(ERROR_CAP)
        for i in range(ERROR_CAP):
            out_fracs[i] = np.sum(array < i)/float(len(array))
        plt.plot(out_fracs, linewidth=2)

    plt.title("Accuracies on " + DATASET_NAME + " dataset", fontsize=16)
    plt.grid()
    plt.xlabel("Registration error threshold [px]", fontsize=14)
    plt.ylabel("Prob. of correct registration [-]", fontsize=14)
    # plt.xscale("log")
    plt.xlim([0, 100])
    plt.ylim([0.6, 1.001])
    plt.legend(names, loc=4, fontsize=16)
    plt.savefig("./accuracy_" + DATASET_NAME + ".eps")
    # plt.show()


def plot_cutout_region(img_path1: str, img_path2: str, region1: [[int], [int]], region2: [[int], [int]]):
    img = image.imread(img_path1)
    img2 = image.imread(img_path2)
    f, axarr = plt.subplots(3)
    axarr[0].imshow(img, aspect="auto")
    axarr[0].axvspan(0, region1[0][0], color='red', alpha=0.5)
    axarr[0].axvspan(region1[0][1], region1[1][0], color='red', alpha=0.5)
    axarr[0].axvspan(region1[1][1], 511,  color='red', alpha=0.5)
    axarr[0].title.set_text("Cutout restriction for coarse dataset")
    axarr[1].imshow(img, aspect="auto")
    axarr[1].axvspan(region2[0][1], region2[1][0], color='red', alpha=0.5)
    rect = patches.Rectangle((440, 0), 40, 40, linewidth=1, edgecolor='k', facecolor='k')
    axarr[1].add_patch(rect)
    axarr[1].title.set_text("Cutout restriction for rectified Nordland dataset")
    print(img2.shape)
    img2 = img2[:img.shape[0], :img.shape[1], :]
    axarr[2].imshow(img2, aspect="auto")
    # axarr[2].axvspan(0, 512, color='green', alpha=0.5)
    axarr[2].title.set_text("Cutout restriction for rectified UTBM robotcar dataset")
    f.tight_layout()
    plt.savefig("./cutouts.png")
    plt.close()


def plot_annotation_ambiguity(img_path1: str, img_path2: str):
    img1 = image.imread(img_path1)
    img2 = image.imread(img_path2)
    f, axarr = plt.subplots(2)
    axarr[0].imshow(img1, aspect="auto")
    axarr[1].imshow(img2, aspect="auto")
    axarr[0].axvline(x=390, ymin=0, ymax=420, c="b")
    axarr[1].axvline(x=400, ymin=0, ymax=420, c="b")
    axarr[0].axvline(x=750, ymin=0, ymax=420, c="r")
    axarr[1].axvline(x=715, ymin=0, ymax=420, c="r")

    f.suptitle("Annotation ambiguity")
    f.tight_layout()
    plt.savefig("./ambiguity.png")
    plt.close()


def plot_features(dataset_path="/home/zdeeno/Documents/Datasets/grief_jpg"):
    from torchvision.io import read_image
    import torch as t
    df = pd.read_csv(os.path.join(dataset_path, "annotation.csv"))
    entries = {"stromovka": [{}], "planetarium": [{} for _ in range(11)],
               "carlevaris": [{}], "michigan": [{} for _ in range(11)]}
    for entry in df.iterrows():
        json_str = entry[1]["meta_info"].replace("'", "\"")
        entry_dict = json.loads(json_str)
        dataset_name = entry_dict["dataset"]
        # this is done for first against all annotation
        if entry_dict["season"] == "":
            season1 = entry_dict["season1"]
            season2 = entry_dict["season2"]
        else:
            season1 = "season_00"
            season2 = "season_01"
        img_idx = int(entry_dict["place"])
        kp_dict1 = json.loads(entry[1]["kp-1"].replace("'", "\""))
        kp_dict2 = json.loads(entry[1]["kp-2"].replace("'", "\""))
        kps1 = []
        kps2 = []
        w, h = None, None
        for kp1, kp2 in zip(kp_dict1, kp_dict2):
            w, h = kp1["original_width"], kp1["original_height"]
            kps1.append((int(kp1["x"]/100 * w), int(kp1["y"]/100 * h)))
            kps2.append((int(kp2["x"]/100 * w), int(kp2["y"]/100 * h)))
        img1 = read_image(os.path.join(dataset_path, dataset_name, season1, str(img_idx).zfill(9)) + ".jpg")
        img2 = read_image(os.path.join(dataset_path, dataset_name, season2, str(img_idx).zfill(9)) + ".jpg")
        big_img = t.cat([img1, img2], dim=1).permute(1, 2, 0).numpy()
        plt.figure()
        plt.imshow(big_img)
        for i in range(len(kps1)):
            plt.plot((kps1[i][0], kps2[i][0]), (kps1[i][1], kps2[i][1] + h))
        img_name = dataset_name + "_" + season1[-2:] + season2[-2:] + "_" + str(img_idx) + ".png"

        plt.savefig(os.path.join("results_annotation", img_name))
        plt.close()
    return entries


def summary(file_path):
    for f in file_path:
        array = np.genfromtxt(f, delimiter=',')  # - diff
        mean_err = np.mean(abs(array))
        acc = np.sum(abs(array) < 32) / array.size
        print(array.size)
        print(mean_err, acc)


if __name__ == '__main__':
    # plot_annotation_ambiguity("/home/zdeeno/Documents/Datasets/grief_jpg/carlevaris/season_00/000000475.jpg",
    #                           "/home/zdeeno/Documents/Datasets/grief_jpg/carlevaris/season_01/000000475.jpg")
    # plot_cutout_region("/home/zdeeno/Documents/Datasets/nordland_rectified/spring/008409.png", [[80, 180], [511 - 180, 511 - 80]], [[0, 180], [511 - 180, 511]])
    error_distribution([# "/home/zdeeno/Documents/Work/GRIEF/results/grief_errors.csv",
                        ["/home/zdeeno/Documents/Work/GRIEF/results/grief_errors_" + DATASET_PLOT.lower() + ".csv", True],
                        ["/home/zdeeno/Documents/Work/GRIEF/results/sift_errors_" + DATASET_PLOT.lower() + ".csv", True],
                        # "/home/zdeeno/Documents/Work/alignment/results_siam/eval_model_70/" + DATASET_PLOT.lower() + "_errors.csv",
                        ["/home/zdeeno/Documents/Work/alignment/results_siam/eval_model_eunord/" + DATASET_PLOT.lower() + "_errors.csv", False],
                        ["/home/zdeeno/Documents/Work/SuperPointPretrainedNetwork/superpixel_errors_" + DATASET_PLOT.lower(), True]],
                        ["grief", "sift", "siamese", "superpoint"])

    # error_distribution([# "/home/zdeeno/Documents/Work/GRIEF/results/grief_errors.csv",
    #                     "/home/zdeeno/Documents/Work/alignment/results_siam/eval_model_20/" + DATASET_PLOT.lower() + "_errors.csv",
    #                     "/home/zdeeno/Documents/Work/alignment/results_siam/eval_model_50/" + DATASET_PLOT.lower() + "_errors.csv",
    #                     "/home/zdeeno/Documents/Work/alignment/results_siam/eval_model_60/" + DATASET_PLOT.lower() + "_errors.csv",
    #                     "/home/zdeeno/Documents/Work/alignment/results_siam/eval_model_70/" + DATASET_PLOT.lower() + "_errors.csv"],
    #                     # "/home/zdeeno/Documents/Work/SuperPointPretrainedNetwork/superpixel_errors_" + DATASET_PLOT.lower()],
    #                      ["20", "50", "60", "70"])
    # plot_features()
    # plot_cutout_region("/home/zdeeno/Documents/Datasets/nordland_rectified/spring/008409.png",
    #                    "/home/zdeeno/Documents/Datasets/eulongterm_rectified/20180716/000100.png",
    #                    [[80, 180], [511 - 180, 511 - 80]],
    #                    [[0, 180], [511 - 180, 511]])
    # summary(["/home/zdeeno/Documents/Work/alignment/results_siam/eval_model_eunord/stromovka_errors.csv",
    #          "/home/zdeeno/Documents/Work/alignment/results_siam/eval_model_eunord/planetarium_errors.csv",
    #          "/home/zdeeno/Documents/Work/alignment/results_siam/eval_model_eunord/cestlice_errors.csv",
    #          "/home/zdeeno/Documents/Work/alignment/results_siam/eval_model_eunord/carlevaris_errors.csv"])