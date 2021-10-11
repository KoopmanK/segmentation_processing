# -*- coding: utf-8 -*-
""" 
Adapted by K. Koopman from WMH segmentation challenge from Hugo Kuijf. 

original: https://github.com/hjkuijf/wmhchallenge/blob/master/evaluation.py
"""


import difflib
import numpy as np
import os
import SimpleITK as sitk
import scipy.spatial
import glob
import json
import tqdm
from pathlib import Path
import pandas as pd

def create_metric_kuijf(experiment_path, train=False, save=True):
    """[Metrics calculated in Kuijf (Utrecht UMC) lesion segmentation challenge. 
    Code slightly adapted. Otherwise the same. ]

    Args:
        experiment_path ([str]): [path to experiment, e.g. HOME:/experiments/experiment_001/]
        train (bool, optional): [Which folder to review. If train --> labelsTr. Otherwise labelsTs]. Defaults to False.
        save (bool, optional): [save as .xlsx to experiment_path]. Defaults to True.

    Returns:
        [dict]: [Keys: patient number, sub-dictionaries with dsc, h95 (mm), avd (%), lesion detection / recall, lesion F1]
    """

    if train:
        testDir = experiment_path+'/4_predictions/labelsTr/'
        participantDir = experiment_path+'/4_predictions/segmentationsTr/'
    else:
        testDir = experiment_path+'/4_predictions/labelsTs/'
        participantDir = experiment_path+'/4_predictions/segmentationsTs/'

    d = {}

    for file in tqdm.tqdm(glob.glob(participantDir + "/*.nii.gz"),  position=0, leave=True):
        filename = Path(file).name
        patientname = filename[:-7]

        testImage, resultImage = getImages(os.path.join(testDir, file[-16:]), file)
        if testImage.GetSize()!= resultImage.GetSize():
            print('Shapes dont match: ', file)
            continue

        dsc = getDSC(testImage, resultImage)
        h95 = getHausdorff(testImage,resultImage)
        avd = getAVD(testImage, resultImage)
        recall, f1 = getLesionDetection(testImage, resultImage)

        d[patientname] = {}
        d[patientname]["dsc"] = dsc
        d[patientname]["h95 (mm)"] = h95
        d[patientname]["avd (%)"] = avd
        d[patientname]["lesion detection / recall"] = recall
        d[patientname]["lesion F1"] = f1

   
    if save:
        df = pd.DataFrame.from_dict({i: d[i] for i in d.keys()}, orient='index')
        df.to_excel(experiment_path+"/metrics_Kuijf.xlsx")
    
    return d

def getImages(testFilename, resultFilename):
    """Return the test and result images, thresholded and non-WMH masked."""
    testImage   = sitk.ReadImage(testFilename)
    resultImage = sitk.ReadImage(resultFilename)
    
    # Get meta data from the test-image, needed for some sitk methods that check this
    if testImage.GetSize()==resultImage.GetSize():
        resultImage.CopyInformation(testImage)
    
    return testImage, resultImage

def getResultFilename(participantDir):
    """Find the filename of the result image.
    
    This should be result.nii.gz or result.nii. If these files are not present,
    it tries to find the closest filename."""
    files = os.listdir(participantDir)
    
    if not files:
        raise Exception("No results in "+ participantDir)
    
    resultFilename = None
    if '*.nii.gz' in files:
        resultFilename = os.path.join(participantDir, 'patient01.nii.gz')
    elif 'result.nii' in files:
        resultFilename = os.path.join(participantDir, 'result.nii')
    else:
        # Find the filename that is closest to 'result.nii.gz'
        maxRatio = -1
        for f in files:
            currentRatio = difflib.SequenceMatcher(a = f, b = 'result.nii.gz').ratio()
            
            if currentRatio > maxRatio:
                resultFilename = os.path.join(participantDir, f)
                maxRatio = currentRatio
                
    return resultFilename
    
    
def getDSC(testImage, resultImage):    
    """Compute the Dice Similarity Coefficient."""
    testArray   = sitk.GetArrayFromImage(testImage).flatten()
    resultArray = sitk.GetArrayFromImage(resultImage).flatten()
    
    # similarity = 1.0 - dissimilarity
    return 1.0 - scipy.spatial.distance.dice(testArray, resultArray) 
    

def getHausdorff(testImage, resultImage):
    """Compute the Hausdorff distance."""
    
    # Hausdorff distance is only defined when something is detected
    resultStatistics = sitk.StatisticsImageFilter()
    resultStatistics.Execute(resultImage)
    if resultStatistics.GetSum() == 0:
        return float('nan')
        
    # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 2D
    eTestImage   = sitk.BinaryErode(testImage, (1,1,0) )
    eResultImage = sitk.BinaryErode(resultImage, (1,1,0) )
    
    hTestImage   = sitk.Subtract(testImage, eTestImage)
    hResultImage = sitk.Subtract(resultImage, eResultImage)    
    
    hTestArray   = sitk.GetArrayFromImage(hTestImage)
    hResultArray = sitk.GetArrayFromImage(hResultImage)   
        
    # testImage.TransformIndexToPhysicalPoint converts (xyz) to world coordinates (in mm)
    testCoordinates   = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hTestArray) ))]
    resultCoordinates = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hResultArray) ))]
        
            
    # Use a kd-tree for fast spatial search
    def getDistancesFromAtoB(a, b):    
        kdTree = scipy.spatial.KDTree(a, leafsize=100)
        return kdTree.query(b, k=1, eps=0, p=2)[0]
    
    # Compute distances from test to result; and result to test
    dTestToResult = getDistancesFromAtoB(testCoordinates, resultCoordinates)
    dResultToTest = getDistancesFromAtoB(resultCoordinates, testCoordinates)    
    
    return max(np.percentile(dTestToResult, 95), np.percentile(dResultToTest, 95))
    
    
def getLesionDetection(testImage, resultImage):    
    """Lesion detection metrics, both recall and F1."""
    
    # Connected components will give the background label 0, so subtract 1 from all results
    ccFilter = sitk.ConnectedComponentImageFilter()    
    ccFilter.SetFullyConnected(True)
    
    # Connected components on the test image, to determine the number of true WMH.
    # And to get the overlap between detected voxels and true WMH
    ccTest = ccFilter.Execute(testImage)    
    lResult = sitk.Multiply(ccTest, sitk.Cast(resultImage, sitk.sitkUInt32))
    
    ccTestArray = sitk.GetArrayFromImage(ccTest)
    lResultArray = sitk.GetArrayFromImage(lResult)
    
    # recall = (number of detected WMH) / (number of true WMH) 
    nWMH = len(np.unique(ccTestArray)) - 1
    if nWMH == 0:
        recall = 1.0
    else:
        recall = float(len(np.unique(lResultArray)) - 1) / nWMH
    
    # Connected components of results, to determine number of detected lesions
    ccResult = ccFilter.Execute(resultImage)
    lTest = sitk.Multiply(ccResult, sitk.Cast(testImage, sitk.sitkUInt32))
    
    ccResultArray = sitk.GetArrayFromImage(ccResult)
    lTestArray = sitk.GetArrayFromImage(lTest)
    
    # precision = (number of detections that intersect with WMH) / (number of all detections)
    nDetections = len(np.unique(ccResultArray)) - 1
    if nDetections == 0:
        precision = 1.0
    else:
        precision = float(len(np.unique(lTestArray)) - 1) / nDetections
    
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    
    return recall, f1    

    
def getAVD(testImage, resultImage):   
    """Volume statistics."""
    # Compute statistics of both images
    testStatistics   = sitk.StatisticsImageFilter()
    resultStatistics = sitk.StatisticsImageFilter()
    
    testStatistics.Execute(testImage)
    resultStatistics.Execute(resultImage)
        
    return float(abs(testStatistics.GetSum() - resultStatistics.GetSum())) / float(testStatistics.GetSum()) * 100