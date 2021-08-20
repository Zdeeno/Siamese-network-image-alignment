from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl


mpl.use('Qt5Agg')


def error_distribution(files: [str], names: [str]):
    for file in files:
        array = np.genfromtxt(file, delimiter=',')
        array = np.array(sorted(abs(array)))
        plt.plot(array, np.linspace(0, 1, len(array)))

    plt.title("Accuracy")
    plt.grid()
    plt.xlabel("error [pix]")
    plt.ylabel("inliers [%]")
    plt.xscale("log")
    plt.legend(names)
    plt.show()


if __name__ == '__main__':
    error_distribution(["/home/zdeeno/Documents/Work/GRIEF/results/grief_errors.csv",
                        "/home/zdeeno/Documents/Work/GRIEF/results/sift_errors.csv",
                        "/home/zdeeno/Documents/Work/alignment/results_siam/eval_model_6/errors.csv"],
                       ["grief", "sift", "siam"])
