import os


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
    list_dir_dataset_path.sort()
    for i in list_dir_dataset_path:
        if 'P' in i:
            folder_list.append(i)
            for j in os.listdir(os.path.join(dataset_path,
                                             i)):
                if 'wav' in j:
                    audio_files.append(j)
                    audio_paths.append(os.path.join(dataset_path,
                                                    i,
                                                    j))
                if 'TRANSCRIPT' in j:
                    if 'lock' in j:
                        pass
                    else:
                        transcript_paths.append(os.path.join(dataset_path,
                                                             i,
                                                             j))

    return folder_list, audio_paths, transcript_paths


