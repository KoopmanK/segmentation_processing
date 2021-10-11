"""
Author: K.Koopman

Statistical test between two experiments based on
DR, DSC or MHD / HDD

"""

from tqdm import tqdm
import json
import glob
from scipy import stats
import numpy as np

def compare_experiments(experiment_path_1, experiment_path_2, metric='DSC'):
    """[Statistical test between two experiments. 
    If one experiment is not normally distributed --> Wilcoxon Signed Rank test. Otherwise Student-t test.
    Return lists with metrics.]

    Args:
        experiment_path_1 ([str]): [path to experiment, e.g. HOME:/experiments/experiment_001/]
        experiment_path_2 ([str]): [path to experiment, e.g. HOME:/experiments/experiment_002/]
        metric (str, optional): [which metric to compare: DR, HDD, DSC]. Defaults to 'DSC'.

    Raises:
        ValueError: [Unequal number of subjects. ]

    Returns:
        [list, list]: [Lists of the metrics on which the test is based on. Returned to make further analysis easier.]
    """
    if metric=='DR':    #DR from lesionMetrics, other from Segmentations
        folder1 = experiment_path_1+"/4_predictions/lesionMetricsTs/"
        folder2 = experiment_path_2+"/4_predictions/lesionMetricsTs/"
    else:
        folder1 = experiment_path_1+"/4_predictions/segmentationsTs/"
        folder2 = experiment_path_2+"/4_predictions/segmentationsTs/"

    if metric=='MHD': #both is fine, but defined as HDD in json
        metric='HDD'

    metric_list_1 = create_metric_list(folder1, metric)
    metric_list_2 = create_metric_list(folder2, metric)
    if len(metric_list_1)!=len(metric_list_2):
        raise ValueError('Unequal number of subjects.')

    #test if data is normally distributed
    #determine which test 
    test = determine_test(metric_list_1, metric_list_2)
    test_stats = do_test(test, metric_list_1, metric_list_2)

    p = test_stats.pvalue
    print('\np-value:', p)
    significance(p)
    return metric_list_1, metric_list_2

def create_metric_list(folder, metric):
    metric_list = []
    for file in tqdm(glob.glob(folder+'/*.json'), position=0, leave=True):
        with open(file) as f:
            data = json.load(f)
            metric_list.append(data['Lesion'][metric])
    return metric_list

def determine_test(metric_list_1, metric_list_2):
    shapiro_1 = stats.shapiro(metric_list_1)
    shapiro_2 = stats.shapiro(metric_list_2)

    if shapiro_1.pvalue < 0.05 and shapiro_2.pvalue < 0.05:
        print('\n\nboth not normally distributed --> test Wilcoxon Signed Ranked')
        test = 'Wilcoxon'
    elif shapiro_1.pvalue < 0.05 < shapiro_2.pvalue:
        print('\n\n', 'experiment_1 not normally distributed --> test Wilcoxon Signed Ranked')
        test = 'Wilcoxon'
    elif shapiro_1.pvalue > 0.05 > shapiro_2.pvalue:
        print('\n\n', 'experiment_2 not normally distributed --> test Wilcoxon Signed Ranked')
        test = 'Wilcoxon'
    else:
        print('\n\nnormal distribution assumed --> student t test')
        test = 'student_t'
    return test

def do_test(test, metric_list_1, metric_list_2):
    global test_stats
    if test=='student_t':
        test_stats = stats.ttest_rel(metric_list_1, metric_list_2)
    elif test=='Wilcoxon':
        test_stats = stats.wilcoxon(metric_list_1, metric_list_2)
    else:
        print('test', test,  ' not recognized') #change to assert
    return test_stats    

def significance(p):
    if p!=p:
        print('files not loaded OR too few examples')
    elif p < 0.05:
        print('significant difference')
    else:
        print('no significant difference')