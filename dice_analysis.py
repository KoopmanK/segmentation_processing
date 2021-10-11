# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 13:45:29 2020

@author: 320009149
"""

import numpy as np
import glob
from scipy import ndimage as nd
import os
import nibabel as nib
import matplotlib.pyplot as plt
import copy
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

def dice_boxplot(experiment_path, train=False):
    """Boxplot with x-axis volume interval and y-axis DSC.
    Compares labels from 2_processed with 4_predictions. 
    Created by Nick Fläschner, Philips Research Hamburg. 

    Args:
        experiment_path ([str]): [path to experiment, e.g. HOME:/experiments/experiment_001/]
        train (bool, optional): [Which folder to review. If train --> labelsTr. Otherwise labelsTs]. Defaults to False.
    """
    if train:
        PATH_label = experiment_path+'/4_predictions/labelsTr/'
        PATH_seg = experiment_path+'/4_predictions/segmentationsTr/'
    else:
        PATH_label = experiment_path+'/4_predictions/labelsTs/'
        PATH_seg = experiment_path+'/4_predictions/segmentationsTs/'
   
    volumes = []
    dices = []  
    for file in tqdm(glob.glob(PATH_label+"/*.nii.gz"), position=0, leave=True):
        gt_data, pixdim = load_image_data(file)
        seg_data, _ = load_image_data(PATH_seg+Path(file).name)
        
        gt_labels = nd.label(gt_data)[0]
        seg_labels = nd.label(seg_data)[0]

        max_val=0
        max_label=0
        for gt_label in np.unique(gt_labels):
            if np.sum(gt_labels==gt_label)>max_val:
                max_val=np.sum(gt_labels==gt_label)
                max_label=gt_label

        for label in tqdm(range(0,np.max(gt_labels)+1), position=0, leave=True):
            if label!=max_label:
                if seg_data.shape!=gt_data.shape:
                    print('shapes dont match')
                    break
                mask = gt_labels==label
                vol_gt = np.sum(mask==True)
                volumes.append(vol_gt)
                overlap = seg_data[mask]==1
                possible_labels = np.unique(seg_labels[mask])
                vol_seg = 0
                for possible_label in possible_labels:
                    if possible_label!=0:
                        vol_seg+=np.sum(seg_labels==possible_label)
                dsc = 2*np.sum(overlap)/(vol_gt+vol_seg)
                dices.append(dsc)        

    volume_per_voxel = pixdim.prod()*10**-3
    real_volumes = volume_per_voxel*np.array(volumes)

    dsc = np.array(copy.copy(dices))

    bins1=np.linspace(0,0.01,11)
    bins2=np.linspace(0.02,0.1,9)
    bins3=np.linspace(0.2,0.5,4)

    xtickdist=1
    bins = np.hstack([bins1,bins2,bins3,100]) 

    data=[]
    labels=[]
    for i in range(0,len(bins)-1):
        selection = (real_volumes>=bins[i]).astype(int) + (real_volumes<bins[i+1]).astype(int)
        # labels.append(((bins[i]+bins[i+1])/2).astype(int))
        number=len(dsc[selection==2])
        labels.append(f'{np.round(bins[i],4)} to {np.round(bins[i+1],4)} : {number}')
        data.append(dsc[selection==2])
            
    labels[-1]=f">=0.5 : {number}"
    ax = sns.boxplot(data=np.array(data, dtype=object),color="tab:blue")  
    ax.set_xticks(list(range(len(bins)-1))[0::xtickdist])
    ax.set_xticklabels(labels[0::xtickdist])
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    ax.set_xlabel("lesion volume [ml]")
    ax.set_ylabel("DSC")
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(experiment_path, "volume_vs_dice.png"),dpi=600)
    plt.close()

def load_image_data(file):
    image = nib.load(file)
    pixdim = image.header['pixdim'][1:4]
    image_data = np.array(image.dataobj)
    return image_data, pixdim
