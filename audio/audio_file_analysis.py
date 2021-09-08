import psutil
import os
from shutil import copyfile, rmtree
import sys
from utils import utilities, file_analysis as fa, plotter
import numpy as np
import librosa
import h5py
from audio import audio_feature_extractor
import pickle
import gc
import math
from config_files import config, config_process, create_folder_name


def modify_audio_file(data, timings, sr, mode=False):
    """
    Function to remove segments from an audio file by creating a new audio
    file containing audio segments from the onset/offset time markers
    specified in the 'timings' variable

    Inputs
        data: numpy.array - The audio data to modify
        timings: list - The onset and offset time markers
        sr: int - The original sampling rate of the audio
        mode: bool - Set True if only considering the background information
              for the audio signal (this is the opening of the audio to the
              point where the first interaction begins)

    Output
        updateed_audio: numpy.array - The updated audio file
    """
    timings = np.array(timings, float)
    samples = timings * sr
    samples = np.array(samples, int)
    pointer = 0
    if mode:
        updated_audio = data[0:samples[0][1]]
    else:
        for i in samples:
            if pointer == 0:
                updated_audio = data[i[0]:i[1]]
                pointer += 1
            else:
                updated_audio = np.hstack((updated_audio, data[i[0]:i[1]]))

    return updated_audio


def max_min_values(current_directory, win_size, hop_size, audio_paths,
                   on_off_times, mode_for_background):
    """
    Calculates the maximum and minimum length of audio in the dataset

    Inputs
        current_directory: str - The location of the current workspace
        win_size: int - The length of the window function
        hop_size: int - The gap between windows passing over the audio
        audio_paths: list - Locations of the audio data to be used
        on_off_times: list - Time markers to extract specific sections of audio
        mode_for_background: bool - Set True for keeping the background
                             information only

    Outputs
        max_value: int - The longest file in the database
        min_value: int - The shortest file in the database
        sample_rate: int - The original sampling rate of the audio
        total_windows_in_file_max: int - The largest file in sampled windows
        total_windows_in_file_min: int - The shortest file in sampled windows
        output_data: numpy.array - Holds meta information collected from each
                     file such as sample rate, number of samples, time in
                     minutes, and folder_number
    """

    print('Processing Audio Files\n')
    max_value = 0
    min_value = 1e20
    output_data = np.zeros((len(audio_paths), 4))
    for iterator, filename in enumerate(audio_paths):
        print(f"iterator is {iterator}, and filename is {filename}")
        audio_data, sample_rate = librosa.load(filename, sr=None)
        mod_audio = modify_audio_file(audio_data,
                                      on_off_times[iterator],
                                      sample_rate,
                                      mode_for_background)
        number_samples = int(mod_audio.shape[0])
        time_in_mins = number_samples / (sample_rate * 60)

        if sys.platform == 'win32':
            folder_name = filename.split('\\')[-2]
        else:
            folder_name = filename.split('/')[-2]

        path = os.path.join(current_directory, config.FEATURE_FOLDERS[0],
                            folder_name + '_audio_data.npy')
        # Save the the audio_data
        np.save(path, mod_audio)
        folder_number = int(folder_name[0:3])

        if audio_data.ndim > 1:
            input('2 Channels were detected')

        if number_samples > max_value:
            max_value = number_samples
        if number_samples < min_value:
            min_value = number_samples

        output_data[iterator, :] = [sample_rate, number_samples,
                                    time_in_mins, folder_number]

    path = os.path.join(current_directory, 'meta_data')
    np.save(path, output_data)

    # To calculate the STFT the following is used:
    # output = ((audio - window) // hop) + 1
    # If padding=True is used, both the start and end are padded by the hop
    # length.
    # output = (audio // hop) + 1
    total_windows_in_file_max = (max_value + (hop_size * 2)) - win_size
    total_windows_in_file_max = (total_windows_in_file_max // hop_size) + 1
    total_windows_in_file_min = (min_value + (hop_size * 2)) - win_size
    total_windows_in_file_min = (total_windows_in_file_min // hop_size) + 1

    return max_value, min_value, sample_rate, total_windows_in_file_max, \
           total_windows_in_file_min, output_data


def create_database(labels, sample_rate, total_windows_in_file_max, max_value,
                    current_directory, features_exp, win_size, hop_size, snv,
                    freq_bins, main_logger, whole_train, gender='-'):
    """
    Creates a database of extracted features from the raw input data such as
    text or audio. The database contains metadata such as folder, class,
    score, and the index of each respective file.

    Inputs
        labels: list - Holds meta data for database including folder, class,
                score, and index
        sample_rate: int - The original sampling rate of the database
        total_windows_in_file_max: int - The longest file in the database in
                                   terms of windowed samples
        current_directory: str - The location of the save directory
        features_exp: str - The type of features to be extracted
        win_size: int - The length of the window to be passed over the audio
        hop_size: int - The gap between the windows passing over the audio
        freq_bins: int - The number of frequency bins to split the features
                   into, for example, features_exp=logmel - freq_bins=64 would
                   result in an audio signal that takes shape of [freq_bins,
                   time]
        main_logger: logger - Records important information
        whole_train: bool - Set True to convert all files to the maximum
                     length found in the database

    Output
        num_samples_feature: list - Records the length of every file to be
                             recorded in another function along with summary
                             data
    """
    # Everything in this h5 dataset is in numerical order. All the
    # folders, classes and features will be written in ascending order.
    # The train, dev and test labels will provide the shuffling
    datatype = h5py.special_dtype(vlen=np.float32)
    if gender == 'f' or gender == 'm':
        h5file = h5py.File(os.path.join(current_directory,
                                        f"complete_database_{gender}.h5"), 'w')
    else:
        h5file = h5py.File(os.path.join(current_directory,
                                        f"complete_database.h5"), 'w')
    num_files = len(labels[0])
    h5file.create_dataset(name='folder',
                          data=[0] * num_files,
                          dtype=np.int16)
    h5file.create_dataset(name='class',
                          data=[0] * num_files,
                          dtype=np.int8)
    h5file.create_dataset(name='score',
                          data=[0] * num_files,
                          dtype=np.int8)
    h5file.create_dataset(name='gender',
                          data=[0] * num_files,
                          dtype=np.int8)
    h5file.create_dataset(name='index',
                          data=[0] * num_files,
                          dtype=np.int16)
    h5file.create_dataset(name='features',
                          compression="gzip",
                          shape=(num_files, 1),
                          maxshape=(num_files, None),
                          dtype=datatype)

    print('\n(Padding Audio) and Creating Spectrogram\n')
    fmin = config.FMIN
    fmax = config.FMAX
    window_func = config.WINDOW_FUNC

    # Feature extractor
    if features_exp == 'logmel' or features_exp == 'mel':
        if features_exp == 'mel':
            log = False
        else:
            log = True
        feature_extractor = audio_feature_extractor.LogMelExtractor(
            sample_rate=sample_rate, window_size=win_size, hop_size=hop_size,
            mel_bins=freq_bins, fmin=fmin, fmax=fmax, window_func=window_func,
            log=log, snv=snv)

    # Loop through all the files and get the sampling rate, number of
    # samples and the time in minutes
    num_samples_feature = []
    for pointer, folder in enumerate(labels[0]):
        audio_file_path = os.path.join(current_directory,
                                       config.FEATURE_FOLDERS[0],
                                       str(folder) + '_P_audio_data.npy')
        updated_file = np.load(audio_file_path)

        # Log mel spectrogram
        if features_exp == 'logmel' or features_exp == 'mel':
            mel_spec = feature_extractor.transform(updated_file)
            plotter.save_plain_plot(current_directory, str(folder), mel_spec,
                                    features_exp)

            print('Folder Name: ', folder, ' dimensions are: ', mel_spec.shape[
                0], mel_spec.shape[1])
            main_logger.info(
                f"Successfully created {features_exp} spectrogram for the "
                f"audio file at: {folder}, it's dimensions "
                f"are: {mel_spec.shape[0]}, "
                f"{mel_spec.shape[1]}")
            height, width = mel_spec.shape
            num_samples_feature.append(width)
            if whole_train:
                length = mel_spec.shape[-1]
                if length < total_windows_in_file_max:
                    diff = int(total_windows_in_file_max - length)
                    mel_spec = np.hstack((mel_spec, np.zeros((freq_bins,
                                                              diff))))
                height, width = mel_spec.shape
            new_length = width * height
            feat_reshaped = np.reshape(mel_spec, new_length)
        elif features_exp == 'spectrogram':
            feat = audio_feature_extractor.sepctrogram(updated_file,
                                                       win_size, hop_size, True,
                                                       window_func, snv)
            plotter.save_plain_plot(current_directory, str(folder), feat,
                                    features_exp)
            print('Folder Name: ', folder, ' dimensions are: ',
                  feat.shape[0], feat.shape[1])
            height, width = feat.shape
            num_samples_feature.append(width)
            if whole_train:
                length = feat.shape[-1]
                if length < total_windows_in_file_max:
                    diff = int(total_windows_in_file_max - length)
                    feat = np.hstack((feat, np.zeros((freq_bins, diff))))
                height, width = feat.shape
            new_length = width * height
            feat_reshaped = np.reshape(feat, new_length)

            main_logger.info(
                f"Successfully created spectrogram for the "
                f"audio file at: {folder}, it's dimensions "
                f"are: {feat.shape[0]}, {feat.shape[1]}")
        elif features_exp.lower() == 'mfcc' or features_exp.lower() == \
                'mfcc_concat':
            mfcc = audio_feature_extractor.mfcc(updated_file, sample_rate,
                                                freq_bins, win_size, hop_size,
                                                window_func, snv)
            print('Folder Name: ', folder, ' dimensions are: ', mfcc.shape[
                0], mfcc.shape[1])
            main_logger.info(
                f"Successfully created MFCC for the audio file at: {folder}, "
                f"it's dimensions are: {mfcc.shape[0]}, {mfcc.shape[1]}")
            height, width = mfcc.shape
            num_samples_feature.append(width)
            if whole_train:
                length = mfcc.shape[-1]
                if length < total_windows_in_file_max:
                    diff = int(total_windows_in_file_max - length)
                    mfcc = np.hstack((mfcc, np.zeros((freq_bins, diff))))
                height, width = mfcc.shape
            new_length = width * height
            feat_reshaped = np.reshape(mfcc, new_length)
        elif features_exp == 'raw':
            feat_reshaped = updated_file
            if snv:
                feat_reshaped = \
                    audio_feature_extractor.standard_normal_variate(feat_reshaped)
            if whole_train:
                length = feat_reshaped.shape[0]
                if length < max_value:
                    diff = int(max_value - length)
                    feat_reshaped = np.hstack((feat_reshaped, np.zeros(diff)))
            num_samples_feature.append(feat_reshaped.shape[-1])

        clss = labels[1][pointer]
        gen = labels[3][pointer]
        if math.isnan(labels[2][pointer]):
            scre = -1
        else:
            scre = labels[2][pointer]

        h5file['features'][pointer] = feat_reshaped
        h5file['folder'][pointer] = folder
        if folder in config_process.wrong_labels:
            h5file['class'][pointer] = config_process.wrong_labels[folder]
        else:
            h5file['class'][pointer] = clss
        h5file['score'][pointer] = scre
        h5file['gender'][pointer] = gen
        h5file['index'][pointer] = pointer

        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss
        memory = memory / 1e9
        if memory > 13:
            gc.collect()

        print(f"This is the value of the pointer, {pointer}")

    h5file.close()
    return num_samples_feature


def process_organise_data(main_logger,
                          current_directory):
    """
    Top level function to process the dataset by filtering out the virtual
    agent's speech from the audio files and determine the meta information
    such as the folder, class, score, index, longest and shortest files
    including the lengths in terms of samples. It then computes the features
    specified for the experiment and saves the results in a database ready
    for model training.

    Inputs
        main_logger: logger - Records the important information
        current_directory: str - The location of the features folder
    """
    dataset_path = config.DATASET
    mode_for_background = config.EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND']
    remove_background = config.EXPERIMENT_DETAILS['REMOVE_BACKGROUND']
    features_exp = config.EXPERIMENT_DETAILS['FEATURE_EXP']
    workspace_files_dir = config.WORKSPACE_FILES_DIR
    win_size = config.EXPERIMENT_DETAILS['WINDOW_SIZE']
    hop_size = config.HOP_SIZE
    freq_bins = config.EXPERIMENT_DETAILS['FREQ_BINS']
    whole_train = config.EXPERIMENT_DETAILS['WHOLE_TRAIN']
    snv = config.EXPERIMENT_DETAILS['SNV']

    main_logger.info(f"The experiment dir is: {features_exp}")
    main_logger.info(f"The feature dir: {current_directory}")
    main_logger.info(f"The dataset dir: {dataset_path}")

    folder_list, audio_paths, transcript_paths = fa.get_meta_data(dataset_path)

    on_off_times = utilities.transcript_file_processing(transcript_paths,
                                                        current_directory,
                                                        mode_for_background,
                                                        remove_background)
    np.save(current_directory+'/on_times.npy', on_off_times)
    main_logger.info(f"The on_off_times are: {on_off_times}")
    # FOR DEBUGGING USE FILES 6:9 IN ORDER TO GET ONE CLASS "1"s
    max_value, min_value, sample_rate, total_windows_in_file_max, \
    total_windows_in_file_min, output_data = max_min_values(
        current_directory, win_size, hop_size, audio_paths, on_off_times,
        mode_for_background)
    print('max_value is: ', max_value, ' number of windows in each file '
                                       'is: ', total_windows_in_file_max)
    print('min_value is: ', min_value, 'number of windows in each file '
                                       'is: ', total_windows_in_file_min)
    main_logger.info(f"The max length (in samples) of the audio is: "
                     f"{max_value}, the minimum is: {min_value}")
    main_logger.info(f"The number of samples after processing spectrogram "
                     f"for the max is {total_windows_in_file_max}, and for "
                     f"the min is {total_windows_in_file_min}")
    utilities.fix_test_files()
    if not os.path.exists(config.COMP_DATASET_PATH):
        if not os.path.exists(config.FULL_TRAIN_SPLIT_PATH):
            utilities.merge_csv(config.TRAIN_SPLIT_PATH, config.DEV_SPLIT_PATH,
                                config.FULL_TRAIN_SPLIT_PATH)
        utilities.merge_csv(config.FULL_TRAIN_SPLIT_PATH,
                            config.TEST_SPLIT_PATH, config.COMP_DATASET_PATH)

    labels = utilities.get_labels_from_dataframe(config.COMP_DATASET_PATH)

    # For debugging purposes
    # l1 = labels[0][0:2]
    # l2 = labels[1][0:2]
    # l3 = labels[2][0:2]
    # l1 = labels[0][0:35]
    # l2 = labels[1][0:35]
    # l3 = labels[2][0:35]
    # l4 = labels[3][0:35]
    # labels = [l1, l2, l3, l4]
    if config.GENDER:
        fin_label = [[[], [], [], []], [[], [], [], []]]
        for i in range(len(labels[0])):
            if labels[-1][i] == 0:
                fin_label[0][0].append(labels[0][i])
                fin_label[0][1].append(labels[1][i])
                fin_label[0][2].append(labels[2][i])
                fin_label[0][3].append(labels[3][i])
            else:
                fin_label[1][0].append(labels[0][i])
                fin_label[1][1].append(labels[1][i])
                fin_label[1][2].append(labels[2][i])
                fin_label[1][3].append(labels[3][i])

        labels = fin_label
        gender = ['f', 'm']
        for i in range(2):
            num_samples_feature = create_database(labels[i], sample_rate,
                                                  total_windows_in_file_max,
                                                  max_value, current_directory,
                                                  features_exp, win_size,
                                                  hop_size, snv, freq_bins,
                                                  main_logger, whole_train,
                                                  gender=gender[i])
    else:
        num_samples_feature = create_database(labels, sample_rate,
                                              total_windows_in_file_max,
                                              max_value, current_directory,
                                              features_exp, win_size,
                                              hop_size, snv, freq_bins,
                                              main_logger, whole_train)

    summary_labels = ['MaxSamples', 'MaxWindows', 'MinSamples', 'MinWindows',
                      'SampleRate', 'NumberFiles', 'ListOfSamples']

    summary_values = [max_value, total_windows_in_file_max, min_value,
                      total_windows_in_file_min, sample_rate, len(labels[0]),
                      num_samples_feature]

    save_path = os.path.join(current_directory, 'summary.pickle')
    with open(save_path, 'wb') as f:
        summary = [summary_labels, summary_values]
        pickle.dump(summary, f)

    copyfile(workspace_files_dir + '/config_files/config.py',
             current_directory + '/config.py')


def startup():
    """
    Starter function to create the working directory and then to process the
    dataset.
    """
    workspace = config.WORKSPACE_MAIN_DIR
    folder_name = create_folder_name.FOLDER_NAME
    current_directory = os.path.join(workspace, folder_name)
    if config.GENDER:
        current_directory = current_directory + '_gen'

    # THIS WILL DELETE EVERYTHING IN THE CURRENT WORKSPACE #
    if os.path.exists(current_directory):
        option = input('A directory at this location exists, do you want '
                       'to delete? ')
        if option == ('y' or 'Y' or 'yes' or 'Yes'):
            rmtree(current_directory, ignore_errors=False, onerror=None)
        else:
            print('Please choose a different path, program will now '
                  'terminate')
            sys.exit()

    os.makedirs(current_directory)
    utilities.create_directories(current_directory, config.FEATURE_FOLDERS)

    main_logger = utilities.setup_logger(current_directory)

    main_logger.info(f"The workspace: {workspace}")

    process_organise_data(main_logger, current_directory)
