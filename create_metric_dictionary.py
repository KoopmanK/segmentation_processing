import os
from tqdm import tqdm
import glob
from pathlib import Path
import json
import pandas as pd

def create_metric_dictionary(experiment_path, train=False, save=True):
    """[Read json files and store results in dictionary. 
    gt volume, seg volume, DR, DSC, HDD. 
    If save, xlsx file is saved. ]

    Args:
        experiment_path ([str]): [path to experiment, e.g. HOME:/experiments/experiment_001/]
        train (bool, optional): [Which folder to review. If train --> labelsTr. Otherwise labelsTs]. Defaults to False.
        save (bool, optional): [save as .xlsx to experiment_path]. Defaults to True.

    Returns:
        [dict]: [Keys: patient number, sub-dictionaries with gt volume, seg volume, DR, DSC and HDD]
    """
    if not os.path.exists(experiment_path):
        raise ValueError(experiment_path, 'does not exist.')

    d = {}  # initialize dictionary

    if train:
        PATH_lesion = experiment_path+'/4_predictions/lesionMetricsTr/training/'
        PATH_segmentation = experiment_path+'/4_predictions/segmentationsTs/training/'
    else:
        PATH_lesion = experiment_path+'/4_predictions/lesionMetricsTs/'
        PATH_segmentation = experiment_path+'/4_predictions/segmentationsTs/'

    for file in tqdm(glob.glob(PATH_lesion+'/*.json'), position=0, leave=True):
        filename = Path(file).name
        patientname = filename[:-13]
        gt_volume = list()
        seg_volume = list()
        with open(file) as f:
            data = json.load(f)
            lesion = data['Lesion']
            for key in data.keys():
                if 'GT' in key:
                    gt_volume.append(data[key]['volume']/1000)
                elif 'DX' in key:
                    seg_volume.append(data[key]['volume']/1000)

        file = PATH_segmentation+filename
        with open(file) as f:
            data = json.load(f)
            segmentation = data['Lesion']

        d[patientname] = {}       #create dictionary per patient name e.g. patient01
        d[patientname]['gt_volume']=sum(gt_volume)
        d[patientname]['seg_volume']=sum(seg_volume)
        d[patientname]['DR']=lesion['DR']
        d[patientname]['DSC']=segmentation['DSC']
        d[patientname]['HDD']=segmentation['HDD']
    
    if save:
        df = pd.DataFrame.from_dict({i: d[i] for i in d.keys()}, orient='index')
        df.to_excel(experiment_path+"/metrics.xlsx")
    
    return d