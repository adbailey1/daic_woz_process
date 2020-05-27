import os
import sys
import numpy as np

# Set to complete to use all the data
# Set to sub to use training/dev sets only
# FEATURE_EXP: logmel or raw or MFCC or MFCC_concat or text
# WHOLE_TRAIN: This setting is for mitigating the variable length of the data
# by zero padding
EXPERIMENT_DETAILS = {'FEATURE_EXP': 'raw',
                      'AUDIO_MODE_IS_CONCAT_NOT_SHORTEN': True,
                      'MAKE_DATASET_EQUAL': False,
                      'FEATURE_DIMENSIONS': 55000,
                      'FREQ_BINS': 1,
                      'SUB_DIR': 'exp_1a',
                      'DATASET_IS_BACKGROUND': False,
                      'CONVERT_TO_IMAGE': False,
                      'WHOLE_TRAIN': False,
                      'WINDOW_SIZE': 1024,
                      'OVERLAP': 50,
                      'SAMPLE_RATE': 16000,
                      'REMOVE_BACKGROUND': True,
                      'SECONDS_TO_SEGMENT': 30}
# Set True to split data into genders
GENDER = True
WINDOW_FUNC = np.hanning(EXPERIMENT_DETAILS['WINDOW_SIZE'])
FMIN = 0
FMAX = EXPERIMENT_DETAILS['SAMPLE_RATE'] / 2
HOP_SIZE = EXPERIMENT_DETAILS['WINDOW_SIZE'] -\
           round(EXPERIMENT_DETAILS['WINDOW_SIZE'] * (EXPERIMENT_DETAILS['OVERLAP'] / 100))

if EXPERIMENT_DETAILS['FEATURE_EXP'] == 'text':
    FEATURE_FOLDERS = None
else:
    FEATURE_FOLDERS = ['audio_data', 'logmel']
EXP_FOLDERS = ['log', 'model', 'condor_logs']

if EXPERIMENT_DETAILS['AUDIO_MODE_IS_CONCAT_NOT_SHORTEN']:
    extension = 'concat'
else:
    extension = 'shorten'
if EXPERIMENT_DETAILS['MAKE_DATASET_EQUAL']:
    data_eq = '_equalSet'
else:
    data_eq = ''
if EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND']:
    bkgnd = '_bkgnd'
else:
    bkgnd = ''

if EXPERIMENT_DETAILS['FEATURE_EXP'] == 'logmel' or EXPERIMENT_DETAILS[
    'FEATURE_EXP'] == 'MFCC' or EXPERIMENT_DETAILS['FEATURE_EXP'] == \
        'MFCC_concat':
    if EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND']:
        FOLDER_NAME = f"BKGND_{EXPERIMENT_DETAILS['FEATURE_EXP']}" \
                      f"_{str(EXPERIMENT_DETAILS['FREQ_BINS'])}_exp"
    elif not EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND'] and \
            EXPERIMENT_DETAILS['REMOVE_BACKGROUND']:
        if EXPERIMENT_DETAILS['WHOLE_TRAIN']:
            FOLDER_NAME = f"{EXPERIMENT_DETAILS['FEATURE_EXP']}" \
                          f"_{str(EXPERIMENT_DETAILS['FREQ_BINS'])}_WIN_" \
                          f"{str(EXPERIMENT_DETAILS['WINDOW_SIZE'])}_OVERLAP_" \
                          f"{str(EXPERIMENT_DETAILS['OVERLAP'])}_WHOLE_expq"
        else:
            FOLDER_NAME = f"{EXPERIMENT_DETAILS['FEATURE_EXP']}" \
                          f"_{str(EXPERIMENT_DETAILS['FREQ_BINS'])}_WIN_" \
                          f"{str(EXPERIMENT_DETAILS['WINDOW_SIZE'])}_OVERLAP_{str(EXPERIMENT_DETAILS['OVERLAP'])}_exp"
    elif not EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND'] and not \
            EXPERIMENT_DETAILS['REMOVE_BACKGROUND']:
        FOLDER_NAME = f"{EXPERIMENT_DETAILS['FEATURE_EXP']}" \
                      f"_{str(EXPERIMENT_DETAILS['MEL_BINS'])}_with_backgnd_exp"
elif EXPERIMENT_DETAILS['FEATURE_EXP'] == 'raw':
    if EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND']:
        FOLDER_NAME = f"BKGND_{EXPERIMENT_DETAILS['FEATURE_EXP']}_exp"
    elif not EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND'] and \
            EXPERIMENT_DETAILS['REMOVE_BACKGROUND']:
        FOLDER_NAME = f"{EXPERIMENT_DETAILS['FEATURE_EXP']}_exp"
    elif not EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND'] and not \
            EXPERIMENT_DETAILS['REMOVE_BACKGROUND']:
        FOLDER_NAME = f"{EXPERIMENT_DETAILS['FEATURE_EXP']}_with_backgnd_exp"
else:
    FOLDER_NAME = f"{EXPERIMENT_DETAILS['FEATURE_EXP']}_exp"
EXP_NAME = f"{extension}{data_eq}{bkgnd}"

if sys.platform == 'win32':
    DATASET = os.path.join('C:', '\\Users', 'Andrew', 'OneDrive', 'DAIC-WOZ')
    WORKSPACE = os.path.join('C:', '\\Users', 'Andrew', 'OneDrive', 'Coding', 'PycharmProjects', 'daic_woz_2')
    TRAIN_SPLIT_PATH = os.path.join(DATASET, 'train_split_Depression_AVEC2017.csv')
    DEV_SPLIT_PATH = os.path.join(DATASET, 'dev_split_Depression_AVEC2017.csv')
    TEST_SPLIT_PATH = os.path.join(DATASET, 'test_split_Depression_AVEC2017.csv')
    FULL_TRAIN_SPLIT_PATH = os.path.join(DATASET, 'full_train_split_Depression_AVEC2017.csv')
    COMP_DATASET_PATH = os.path.join(DATASET, 'complete_Depression_AVEC2017.csv')
elif sys.platform == 'linux' and not os.uname()[1] == 'andrew-ubuntu':
    DATASET = os.path.join('/vol/vssp/datasets/singlevideo01/DAIC-WOZ')
    # set the path of the workspace (where the code is)
    WORKSPACE_FILES_DIR = \
        '/user/HS227/ab01814/pycharm_projects/daic_woz_process'
    # set the path of the workspace (where the models/output will be stored)
    WORKSPACE_MAIN_DIR = '/vol/research/ab01814_res/daic_woz_2'
    TRAIN_SPLIT_PATH = os.path.join(DATASET, 'train_split_Depression_AVEC2017.csv')
    DEV_SPLIT_PATH = os.path.join(DATASET, 'dev_split_Depression_AVEC2017.csv')
    TEST_SPLIT_PATH = os.path.join(DATASET, 'test_split_Depression_AVEC2017.csv')
    FULL_TRAIN_SPLIT_PATH = os.path.join(DATASET, 'full_train_split_Depression_AVEC2017.csv')
    COMP_DATASET_PATH = os.path.join(DATASET, 'complete_Depression_AVEC2017.csv')
elif os.uname()[1] == 'andrew-ubuntu':
    DATASET = '/mnt/6663b3e6-a12f-49e8-b881-421cebf2f8c6/datasets/DAIC-WOZ'
    WORKSPACE_MAIN_DIR = '/mnt/6663b3e6-a12f-49e8-b881-421cebf2f8c6/daic_woz_2'
    WORKSPACE_FILES_DIR = os.path.join('/home', 'andrew', 'PycharmProjects',
                                       'daic_woz_process')
    TRAIN_SPLIT_PATH = os.path.join(DATASET, 'train_split_Depression_AVEC2017.csv')
    DEV_SPLIT_PATH = os.path.join(DATASET, 'dev_split_Depression_AVEC2017.csv')
    TEST_SPLIT_PATH = os.path.join(DATASET, 'test_split_Depression_AVEC2017.csv')
    FULL_TRAIN_SPLIT_PATH = os.path.join(DATASET, 'full_train_split_Depression_AVEC2017.csv')
    COMP_DATASET_PATH = os.path.join(DATASET, 'complete_Depression_AVEC2017.csv')
else:
    DATASET = os.path.join('Users', 'andrewbailey', 'OneDrive', 'DAIC-WOZ')
    WORKSPACE = os.path.join('Users', 'andrewbailey', 'OneDrive', 'Coding', 'PycharmProjects', 'daic_woz_2')
    TRAIN_SPLIT_PATH = os.path.join('Users', 'andrewbailey', 'OneDrive', 'DAIC-WOZ', 'train_split_Depression_AVEC2017.csv')
    DEV_SPLIT_PATH = os.path.join('Users', 'andrewbailey', 'OneDrive', 'DAIC-WOZ', 'dev_split_Depression_AVEC2017.csv')
    TEST_SPLIT_PATH = os.path.join('Users', 'andrewbailey', 'OneDrive', 'DAIC-WOZ', 'test_split_Depression_AVEC2017.csv')
