import os
import numpy as np
from scipy.stats import kurtosis, skew, ttest_rel
from scipy.stats import wilcoxon

base_method = 'grief'
base_filename = base_method + '_errors.csv'

def compareMethods(a, b, t_test=False):
    if t_test:
        print("T-test (alternative: methodA > methodB)")
        print("statistic\t%s" % (ttest_rel(a, b, alternative='greater')[0]))
        print("p\t%s" % (ttest_rel(a, b)[1]))
    else:
        #perform bootstrap resampling
        print("Wilcoxon paired signed-rank test (alternative: methodA > methodB)")
        print("Wilcoxon W\t%s" % (wilcoxon(a, b, alternative='greater')[0]))
        print("Wilcoxon p\t%s" % (wilcoxon(a, b, alternative='greater')[1]))

openfile = open(os.path.join("./eval_errors", base_filename), "r")
base_errors = np.array([float(line[:-1]) for line in openfile.readlines()])
openfile.close()
print("Skew: {0:.2f}, Kurtosis: {0:.2f} for the base method.".format(skew(base_errors), kurtosis(base_errors)))

for method_filename in os.listdir("eval_errors"):
    if method_filename == base_method + '_errors.csv':
        continue

    errors = []
    if method_filename.endswith(".csv"):
        openfile = open(os.path.join("./eval_errors", method_filename), "r")
        errors = [float(line[:-1]) for line in openfile.readlines()]
        openfile.close()
    errors = np.array(errors)
    print("--")
    print("Skew: {0:.2f}, Kurtosis: {0:.2f} for the method.".format(skew(errors), kurtosis(errors)))
    print("Comparing of the methods: %s w/ %s" % ('grief', method_filename[:-11]))
    compareMethods(base_errors, errors)
    #compareMethods(base_errors, errors, t_test=True)

