**Credit**

This repository relates to our work in the EUSIPCO 2021 paper, "Gender Bias in Depression Detection Using Audio Features", https://arxiv.org/abs/2010.15120

**PURPOSE**

The DAIC-WOZ dataset contains many errors and noise (such as interruptions
 during an interview or missing transcript files for the virtual agent). This
  pre-processing framework has been designed to remove these errors to
   provide a cleaner dataset to use for model building and testing. 

List of known errors:
 - Every file contains interactions between the participant and researcher prior to the interview starting which need to be removed.
 - Some interviews (373 and 444) contain long interruptions which should be removed.
 - Some interviews (451, 458, and 480,) are missing the virtual agent's transcriptions.
 - Some interviews (318, 321, 341, and 362) transcription files are out of sync with the audio. 
 - A labelling error was also found for interview 409, whose PHQ-8 score was 10 but the binary value given was 0 rather than 1. 

At this time, only the audio and the transcript files can be processed
 with the visual data to be updated in the future.
  ***UPDATE*** The Text data has not been tested for latest update
  The transcript files
  are converted to Word2Vec but the user can specify which audio features
   to extract - raw audio, log-mel spectrogram, power spectrogram, MFCC. 
      
The output of this framework is a h5py file containing a cleaned, feature
 extracted database ready for experimentation. Due to the nature of h5py, if
  2D features are extracted, they are first flattened before being added, for
   example for a file that is 100 samples long where log-mel spectrogram was
    chosen with the number of mel bins = 64, the resulting file will be 6400
     samples long and will need to be reshaped before use in training. 

To obtain the features, class labels, scores, gender, folders, and corresponding
 indexes for the extracted features use the following:
`h5 = hrpy.File('filename.h5', 'r')
features = h5['features']
labels = h5['class']
scores = h5['score']
folders = h5['folder']
genders = h5['gender']
index = h5['index']`

2 further files will be created, summary.pickle which is a list of lists. The first list contains headers to the corresponding index in the second list. 
The other file is meta_data.npy which holds sample rate, number of samples in current file, corresponding time in minutes, and folder number. 

**SETUP**

This has been designed for Ubuntu (18.04) using Python 3

Install miniconda and load the environment file from environment.yml file

For Linux: 

`conda env create -f environment.yml`


Activate the new environment: `conda activate myenv`
If you have used the `environment.yml` file provided here, the name of the environment will be "daic", therefore:
`conda activate daic`

For textual feature extraction the current framework uses Gensim 
(https://radimrehurek.com/gensim/), current work is on creating an
 alternative using Spacy (https://spacy.io/).

**DATASET**

For this experiment, the DAIC-WOZ dataset is used. This can be obtained
 through The University of Southern California (http://dcapswoz.ict.usc.edu
 /) by signing an agreement form. The dataset is roughly 135GB. 


**EXPERIMENT SETUP**

Before Running an Experiment:
Use the config.py file to set experiment preferences and locations of the code, 
workspace, and dataset directories etc. The main variables of interest are
 found in the dictionary- EXPERIMENT_DETAILS. 

To run the framework, go to the daic_woz_process directory and run:
 `python -m run`
 
Please choose which test file to use in the creation of the features 
database. The 'test_split_Depression_AVEC2017.csv' 
file does not contain depression labels, only ID and Gender. If 
your dataset came with 'full_test_split.csv' file, this should have the 
depression labels. 

To change which file is used in the pre-processing tool, 
change the following command in config_files/config.py from:

`TEST_SPLIT_PATH = TEST_SPLIT_PATH_1`

to 

`TEST_SPLIT_PATH = TEST_SPLIT_PATH_2`