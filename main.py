from create_metric_dictionary import create_metric_dictionary
from eval_processing import evaluate_processing
from evaluation_Kuijf import create_metric_kuijf
from bland_altman import bland_altman
from compare_experiments import compare_experiments
from dice_analysis import dice_boxplot
from volume_histogram import frequency_histogram

#%% Dataset analysis
data_path_OpenMS = '../OpenMS/labels/'#'../OpenMS/labels/'
data_path_JH = '..Johns_Hopkins/labels/'#'../JohnsHopkins/labels 

#images are saved in first path
frequency_histogram(data_path_OpenMS, 'OpenMS')     #plot histogram for this dataset
frequency_histogram(data_path_JH, 'Johns Hopkins')  #""
frequency_histogram(data_path_OpenMS, 'OpenMS', data_path_JH, 'Johns Hopkins')  #plot histogram with both to compare

#%% Experiment analysis
path = 'experiments/'     #define path where experiments are stored
experiment = 'experiment_001'
experiment_path = path+experiment
train = False        #True --> images from imagesTr. False --> images from imagesTs

#compare 1_original with 2_processed
save_processing = True
if save_processing:
    evaluate_processing(experiment_path, train) 

# boxplot DSC per volume
dice_boxplot(experiment_path, train)

# calculate metrics: stored as dictionary with patient numbers
save_excel = True   
metrics = create_metric_dictionary(experiment_path, train, save_excel)
metrics_kuijf = create_metric_kuijf(experiment_path, train, save_excel)

# divide patient numbers on dataset for color coded BA plots
dataset={'OpenMS':['patient00', 'patient00', 'patient00'], 'Johns Hopkins':[]}
for key in metrics.keys():
    if key in dataset['OpenMS']:
        continue
    else:
        dataset['Johns Hopkins'].append(key)
#bland altman plot based on metrics and dataset
bland_altman(experiment_path, metrics, dataset)

#compare two experiments: statistical test
compare_exp = 'experiment_002'
metriclist_1, metriclist_2 = compare_experiments(experiment_path, path+compare_exp, metric='DSC') 