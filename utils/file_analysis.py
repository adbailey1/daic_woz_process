import os
import sys
from zipfile import ZipFile

def get_meta_data(dataset_path):
    """
    Grabs meta data from the dataset including, a list of the folders,
    a list of the audio paths, and a list of the transcription files for all
    the files in the dataset

    Input
        dataset_path: str - Location of the dataset

    Outputs
        folder_list: list - The complete list of folders in the dataset
        audio_paths: list - The complete list of audio locations for the data
        transcript_paths: list - The complete list of locations of the
                          transcriptions
    """
    folder_list = []
    audio_files = []
    audio_paths = []
    transcript_paths = []
    list_dir_dataset_path = os.listdir(dataset_path)
    counter = 0
    for file in list_dir_dataset_path:
        if file.endswith('.zip'):
            if counter == 0:
                print('Converting zip files...')
                counter += 1
            current_file = os.path.join(dataset_path, file)
            new_file = file.split('.')[0]
            try:
                with ZipFile(current_file, 'r') as zipObj:
                    # Extract all the contents of zip file in different directory
                    zipObj.extractall(os.path.join(dataset_path, new_file))
                    os.remove(current_file)
            except:
                print(f"The file {current_file} may not have downloaded "
                      f"correctly, please try re-downloading and running the "
                      f"pre-processing tool again")
                sys.exit()

    list_dir_dataset_path = os.listdir(dataset_path)
    list_dir_dataset_path.sort()
    for i in list_dir_dataset_path:
        if i.endswith('_P'):
            folder_list.append(i)
            for j in os.listdir(os.path.join(dataset_path, i)):
                if 'wav' in j:
                    audio_files.append(j)
                    audio_paths.append(os.path.join(dataset_path, i, j))
                if 'TRANSCRIPT' in j:
                    if 'lock' in j or '._' in j:
                        pass
                    else:
                        transcript_paths.append(os.path.join(dataset_path, i, j))

    return folder_list, audio_paths, transcript_paths


