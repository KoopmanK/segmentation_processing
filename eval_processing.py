"""
Author: K. Koopman

Evaluate two datasets and difference: voxel dimensions, voxel volume, nr lesions, 
min, median, mean, max and total volume. 
"""
from tqdm import tqdm
import glob
import nibabel as nib
from pathlib import Path
import numpy as np
from skimage.measure import label
import pandas as pd
from statistics import median, mean

def evaluate_processing(experiment_path, train=False):
    """[Evaluation of 1_original vs 2_processed folder of an experiment. 
    Saves as .xlsl in experiment_path. 
    - voxel dimensions
    - voxel volume
    - image dimensions
    - number of lesions
    - min, median, mean, max and total volume ]

    Args:
        experiment_path ([type]): [path to experiment, e.g. HOME:/experiments/experiment_001/]
        train (bool, optional): [Which folder to review. If train --> labelsTr. Otherwise labelsTs]. Defaults to False.
    """

    d = {}

    if train:
        PATH_original = experiment_path+'/1_original/labelsTr/'
        PATH_processed = experiment_path+'/2_processed/labelsTr/'
    else:
        PATH_original = experiment_path+'/1_original/labelsTs/'
        PATH_processed = experiment_path+'/2_processed/labelsTs/'

    for file in tqdm(glob.glob(PATH_original+"/*.nii.gz"), position=0, leave=True):
        filename = Path(file).name
        patientname = filename[:-7]
        
        original_data, orig_pixdim = load_image_data(file)
        orig_volume_per_voxel = orig_pixdim.prod()*0.001
        orig_count = voxel_counter(original_data)
        orig_lesion_vol = image_volume(orig_pixdim, orig_count)

        processed_data, proc_pixdim = load_image_data(PATH_processed+Path(filename).name)
        proc_volume_per_voxel = proc_pixdim.prod()*0.001
        proc_count = voxel_counter(processed_data)
        proc_lesion_vol = image_volume(proc_pixdim, proc_count)

        d[patientname] = {}
        prefix = '1_'
        d[patientname][prefix+'voxel dimensions']  = orig_pixdim
        d[patientname][prefix+'voxel volume']      = orig_volume_per_voxel
        d[patientname][prefix+'image dimensions']  = original_data.shape
        d[patientname] = finish_volume_dict(d[patientname], orig_lesion_vol, prefix)
        d[patientname]["_"]                   = " "

        prefix = '2_'
        d[patientname][prefix+'voxel dimensions']  = proc_pixdim
        d[patientname][prefix+'voxel volume']      = proc_volume_per_voxel
        d[patientname][prefix+'image dimensions']  = processed_data.shape
        d[patientname] = finish_volume_dict(d[patientname], proc_lesion_vol, prefix)
        d[patientname]["__"]                  = " "

        prefix = 'difference_'
        d[patientname][prefix+'voxel dimensions']  = tuple(map(lambda i, j: i - j, proc_pixdim, orig_pixdim))
        d[patientname][prefix+'voxel volume']      = proc_volume_per_voxel-orig_volume_per_voxel
        d[patientname][prefix+'image dimensions']  = tuple(map(lambda i, j: i - j, processed_data.shape, original_data.shape))
        d[patientname][prefix+'number of lesions'] = len(proc_lesion_vol)-len(orig_lesion_vol)
        d[patientname][prefix+'min volume']        = min(proc_lesion_vol)-min(orig_lesion_vol)
        d[patientname][prefix+'median volume']     = median(proc_lesion_vol)-median(orig_lesion_vol)
        d[patientname][prefix+'mean volume']       = mean(proc_lesion_vol)-mean(orig_lesion_vol)
        d[patientname][prefix+'max volume']        = max(proc_lesion_vol)-max(orig_lesion_vol)
        d[patientname][prefix+'total volume']      = sum(proc_lesion_vol)-sum(orig_lesion_vol)

    df = pd.DataFrame.from_dict({i: d[i] for i in d.keys()}, orient='index')
    df.to_excel(experiment_path+"/eval_processing.xlsx")

def load_image_data(file):
    image = nib.load(file)
    pixdim = image.header['pixdim'][1:4]
    image_data = np.array(image.dataobj)
    return image_data, pixdim

def voxel_counter(image_data):
    labels = label(image_data)
    _, count = np.unique(labels, return_counts=True)
    return count

def image_volume(pixdim, count):
    vol_per_voxel = pixdim.prod()*0.001
    volume = list(count*vol_per_voxel)
    volume = volume[1:]
    return volume

def finish_volume_dict(patient_dict, volume, prefix):
    patient_dict[prefix+'number of lesions'] = len(volume)
    patient_dict[prefix+'min volume']        = min(volume)
    patient_dict[prefix+'median volume']     = median(volume)
    patient_dict[prefix+'mean volume']       = mean(volume)
    patient_dict[prefix+'max volume']        = max(volume)
    patient_dict[prefix+'total volume']      = sum(volume)
    return patient_dict    
