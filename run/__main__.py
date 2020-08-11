import os
from config_files import config
from audio import audio_file_analysis
from text import text_file_analysis

if __name__ == "__main__":
    """
    This is used to determine whether the textual features should be 
    extracted or the audio features. 
    """
    current_path = os.path.dirname(os.path.realpath(__file__))
    feature_type = config.EXPERIMENT_DETAILS['FEATURE_EXP']
    if feature_type[0:4] == 'text':
        text_file_analysis.startup()
    else:
        audio_file_analysis.startup()

    print('Finished Processing')

