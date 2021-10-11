"""Author: K. Koopman

"""
from numpy.core.fromnumeric import mean, std
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os

def bland_altman(experiment_path, data, datasets={}):
    """[Create Bland Altman plot, saved in experiment_path. 
    ]

    Args:
        experiment_path ([str]): [path to experiment, e.g. HOME:/experiments/experiment_001/]
        data ([str]): [metrics from create_metric_dictionary. Volumes are extracted to create the plot.]
        datasets (dict, optional): [e.g. dataset={'OpenMS':['patient05', 'patient08', 'patient29'], 'Johns Hopkins':[]}
        To color code different datasets, define dict with keys=datasets and value is list with patient names]. Defaults to {}.
    """
    d = {'dataset':[], 'mean':[], 'diff':[]}
    gt = []
    seg = []
    for number in data.keys():        
        gt.append(data[number].get('gt_volume'))
        seg.append(data[number].get('seg_volume'))
        if datasets:    # if datasets with patient number specified, plot different colors
            for name in datasets.keys():
                if number in datasets[name]:
                    d['dataset'].append(name)
        else:           # if not specified, call everything data
            d['dataset'].append('data')

    gt = np.asarray(gt)
    seg = np.asarray(seg)
    mean = np.mean([gt, seg], axis=0)
    diff = seg - gt
    md = np.mean(diff)
    sd = np.std(diff, axis=0)        

    d['mean'].extend(mean)
    d['diff'].extend(diff)

    df = pd.DataFrame(data=d)

    sns.scatterplot(data=df, x='mean', y='diff', hue='dataset')
    plt.axhline(0,            color='gray', linestyle='-')
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')

    plt.title('Bland-Altman Plot')
    plt.xlabel('mean volume (ml)')
    plt.ylabel('difference volume (ml)')
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path, "bland_altman.png"), dpi=600)
    plt.close()