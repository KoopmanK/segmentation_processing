""" 
Author: K. Koopman

Plot volume histograms for dataset and individual patients.
"""

import numpy as np
import os
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from eval_processing import load_image_data, voxel_counter, image_volume
import pandas as pd

def frequency_histogram(data_path, data_name, compare_path="", compare_name=""):
    """[Histogram with x-axis volume intervals and y-axis frequency. 
    For one datasets or two if compare_path and compare_name are stated.
    Plot is stored in 'data_path' in both cases.]

    Args:
        data_path ([str]): [path to data labels, e.g. HOME:/OpenMS/labels/]
        data_name ([str]): [name of dataset, e.g. 'OpenMS']
        compare_path (str, optional): [path to data labels for dataset to compare with. e.g. HOME:/Johns_Hopkins/labels/]. Defaults to "".
        compare_name (str, optional): [name of dataset to compare with, e.g. 'Johns Hopkins']. Defaults to "".
    """
    volume = []
    for file in tqdm(glob.glob(data_path+"/*.nii.gz")+glob.glob(data_path+"/*.nii"), position=0, leave=True):
        volume.extend(create_volume_list(file))
  
    bins1 = np.linspace(0,0.01,11)
    bins2 = np.linspace(0.02,0.1,9)
    bins3 = np.linspace(0.2,0.5,4)
    bins = np.hstack([bins1, bins2, bins3, 100])

    hist_gt, _ = np.histogram(volume, bins)    #number of occurences per bin
    hist_gt_normalized = hist_gt / sum(hist_gt)

    if compare_path:
        compare_volume = []
        for file in tqdm(glob.glob(compare_path+"/*.nii.gz")+glob.glob(compare_path+"/*.nii"), position=0, leave=True):
            compare_volume.extend(create_volume_list(file))
        hist_compare, _ = np.histogram(compare_volume, bins)
        hist_compare_normalized = hist_compare / sum(hist_compare)

        labels = []
        for i in range(0, len(bins)-1):
            labels.append(f'{np.round(bins[i],4)} to {np.round(bins[i+1],4)}')
        labels[-1]=f">=0.5"       

    else:
        labels = []
        for i in range(0, len(bins)-1):
            labels.append(f'{np.round(bins[i],4)} to {np.round(bins[i+1],4)} : {hist_gt[i]}')
        labels[-1]=f">=0.5 : {hist_gt[i]}"
    
    d = {"bin":[], "nr":[], "data":[], "frequency":[]}
    for i in range(len(hist_gt)):        
        d["bin"].append(bins[i+1])
        d["nr"].append(hist_gt[i])
        d["data"].append(data_name)
        d["frequency"].append(hist_gt_normalized[i])
        if compare_path:
            d["bin"].append(bins[i+1])
            d["nr"].append(hist_compare[i])        
            d["data"].append(compare_name)        
            d["frequency"].append(hist_compare_normalized[i])

    df = pd.DataFrame(data=d)

    ax=sns.barplot(data=df, x="bin", y="frequency", hue="data")
    ax.set_xticks(list(range(len(bins)-1))[0::1])
    ax.set_xticklabels(labels[0::1])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel("lesion volume [ml]")
    ax.set_ylabel("frequency")
    plt.tight_layout()
    if compare_path:
        plt.savefig(os.path.join(data_path, "compare_histogram_"+data_name+"_"+compare_name+".png"), bbox_inches="tight", dpi=600)
    else:
        plt.savefig(os.path.join(data_path, "frequency_histogram.png"), bbox_inches="tight", dpi=600)
    plt.close()

def create_volume_list(file):
    data, pixdim = load_image_data(file)
    count = voxel_counter(data)
    vol = image_volume(pixdim, count)
    return vol