import sys
import os
import pickle
import numpy as np
import h5py
import pandas as pd
import argparse
import logging
import logging.handlers
import csv
import shutil
from config_files import config_process, config
from gensim import corpora


def fix_test_files():
    """
    Explicitly fixes issues in both sets of test .csv files (should they exist)
    :return:
    """
    test_file1 = config.TEST_SPLIT_PATH_1
    test_file2 = config.TEST_SPLIT_PATH_2
    train_file = config.TRAIN_SPLIT_PATH

    files = [test_file1, test_file2]
    for i in files:
        if os.path.exists(i):
            print(f"The file does not exist: {i}")
            pass
        else:
            _, b, _, columnsb = meta_data_checker(train_file, i)

            temp = i.split('.')[0] + '_original.csv'
            os.rename(i, temp)
            b = b.sort_values(by=['Participant_ID'])
            b.to_csv(i, index=False)


def meta_data_checker(path1, path2):
    """
    Checks the meta-data from the .csv files from path1 and path2. If the
    headers are different, resolve by using path1 (should be
    train_split_Depression_AVEC2017.csv) as the standard. Also, check and fix
    any wrong labels

    :param path1: Path to the first .csv file
    :param path2: Path to the second .csv file
    :return: Updated dataframes for path1, path2 and their respective column
    headers
    """
    a = pd.read_csv(path1)
    b = pd.read_csv(path2)

    columnsa = list(a)
    columnsb = list(b)
    # The test split doesn't have the same columns as the train/dev
    # Create these extra columns and fill them with -1
    if len(columnsa) > len(columnsb):
        loc_to_indx = {i: header for i, header in enumerate(columnsa)}
        for i in loc_to_indx:
            if i < len(columnsb):
                if loc_to_indx[i] == columnsb[i]:
                    pass
                elif loc_to_indx[i].lower() == columnsb[i].lower():
                    b = b.rename(columns={columnsb[i]: columnsa[i]})
                elif loc_to_indx[i].split('_')[-1] == columnsb[i].split('_')[
                    -1]:
                    b = b.rename(columns={columnsb[i]: columnsa[i]})
                else:
                    try:
                        b.insert(i, loc_to_indx[i], [-1] * b.shape[0])
                        columnsb = list(b)
                    except:
                        b[loc_to_indx[i]] = [-1] * b.shape[0]
            else:
                break
        difference = len(columnsa) - len(columnsb)
        names = columnsa[-difference:]
        h, _ = b.shape
        in_place = [-1] * h
        for i in range(difference):
            b[names[i]] = in_place

    # Only need to do this over dataframe a as the only error currently
    # known is in train_split_Depression_AVEC2017.csv
    for file in config_process.wrong_labels:
        if file in list(a['Participant_ID']):
            location = list(a['Participant_ID']).index(file)
            if a.loc[location, 'PHQ8_Binary'] != config_process.wrong_labels[
                file]:
                a.loc[location, 'PHQ8_Binary'] = config_process.wrong_labels[file]

                temp = path1.split('.')[0] + '_original.csv'
                os.rename(path1, temp)
                a = a.sort_values(by=['Participant_ID'])
                a.to_csv(path1, index=False)
    return a, b, columnsa, columnsb


def transcript_file_processing(transcript_paths, current_dir,
                               mode_for_bkgnd=False, remove_background=True):
    """
    Goes through the transcript files in the dataset and processes them in
    several ways. For the known files that contain errors, config_process,
    these are corrected. The participant and virtual agent's dialogue are
    recorded in order to remove the virtual agent in a later function. This
    also removes the background noise present at the beginning of the
    experiment unless the experiment is to solely work on this. The main
    principle in the processing is to record the onset and offset times of
    each utterance from the participant so these can be extracted from
    audio data and experimented on

    Inputs
        transcript_paths: str - The location of the transcripts
        current_dir: str - The location of the current working directory to
                     save the time signatures for each file
        mode_for_bkgnd: bool - If True, only consider the up to the
                        virtual agent's introduction, this is considered the
                        background
    remove_background: bool - Set True, if the information pre-virtual
                       agent's introduction should be removed

    Output
        on_of_times: list - Record of the participants speech time markers
                     for every file in the dataset.
    """
    on_off_times = []
    # Interruptions during the session
    special_case = config_process.interrupt
    # Misaligned transcript timings
    special_case_3 = config_process.misaligned
    for i in transcript_paths:
        trial = i.split('/')[-2]
        trial = int(trial.split('_')[0])
        with open(i, 'r') as file:
            data = file.readlines()
        ellies_first_intro = 0
        inter = []
        for j, values in enumerate(data):
            file_end = len(data) - 1
            # The headers are in first position
            if j == 0:
                pass
            else:
                # The values in this list are actually strings, not int
                # Create temp which holds onset/offset times and the
                # speaker id
                temp = values.split()[0:3]
                # This corrects misalignment errors
                if trial in special_case_3:
                    if len(temp) == 0:
                        time_start = time_end = 0
                    else:
                        time_start = float(temp[0]) + special_case_3[trial]
                        time_end = float(temp[1]) + special_case_3[trial]
                else:
                    if len(temp) == 0:
                        time_start = time_end = 0
                    else:
                        time_start = float(temp[0])
                        time_end = float(temp[1])
                if len(values) > 1:
                    sync = values.split()[-1]
                else:
                    sync = ''
                if sync == '[sync]' or sync == '[syncing]':
                    sync = True
                else:
                    sync = False
                if len(temp) > 0 and temp[-1] == ('Participant' or
                                                  'participant'):
                    if sync:
                        pass
                    else:
                        if trial in special_case:
                            inter_start = special_case[trial][0]
                            inter_end = special_case[trial][1]
                            if time_start < inter_start < time_end:
                                inter.append([time_start, inter_start - 0.01])
                            elif time_start < inter_end < time_end:
                                inter.append([inter_end + 0.01, time_end])
                            elif inter_start < time_start < inter_end:
                                pass
                            elif inter_start < time_end < inter_end:
                                pass
                            elif time_end < inter_start or time_start > inter_end:
                                inter.append(temp[0:2])
                        else:
                            if 0 < j:
                                prev_val = data[j-1].split()[0:3]
                                if len(prev_val) == 0:
                                    if j - 2 > 0:
                                        prev_val = data[j-2].split()[0:3]
                                    else:
                                        prev_val = ['', '', 'Ellie']
                                if j != file_end:
                                    next_val = data[j+1].split()[0:3]
                                    if len(next_val) == 0:
                                        if j+1 != file_end:
                                            next_val = data[j + 2].split()[0:3]
                                        else:
                                            next_val = ['', '', 'Ellie']
                                else:
                                    next_val = ['', '', 'Ellie']
                                if prev_val[-1] != ('Participant' or
                                                    'participant'):
                                    holding_start = time_start
                                elif prev_val[-1] == ('Participant' or
                                                      'participant'):
                                    pass
                                if next_val[-1] == ('Participant' or
                                                    'participant'):
                                    continue
                                elif next_val[-1] != ('Participant' or
                                                      'participant'):
                                    holding_stop = time_end
                                    inter.append([str(holding_start),
                                                  str(holding_stop)])
                            else:
                                inter.append([str(time_start), str(time_end)])
                elif not temp or temp[-1] == ('Ellie' or 'ellie') and not \
                        mode_for_bkgnd and not sync:
                    pass
                elif temp[-1] == ('Ellie' or 'ellie') and mode_for_bkgnd \
                        and not sync:
                    if ellies_first_intro == 0:
                        inter.append([0, str(time_start - 0.01)])
                        break
                elif temp[-1] == ('Ellie' or 'ellie') and sync:
                    if remove_background or mode_for_bkgnd:
                        pass
                    else:
                        inter.append([str(time_start), str(time_end)])
                        ellies_first_intro = 1
                else:
                    print('Error, Transcript file does not contain '
                          'expected values')
                    print(f"File: {i}, This is from temp: {temp[-1]}")
                    sys.exit()
        on_off_times.append(inter)

    with open(os.path.join(current_dir, 'on_off_times.pickle'), 'wb') as f:
        pickle.dump(on_off_times, f)

        return on_off_times


def transcript_text_processing(transcript_paths,
                               current_dir):
    """
    Goes through the transcript files in the dataset and processes them in
    several ways. For the known files that contain errors, config_process,
    these are corrected. The participant's dialogue is recorded in order to
    convert the words into features such as Word2Vec. This also removes the
    background noise present at the beginning of the experiment. Once the
    participant's utterances have been extracted, for every file, they are
    converted to a corpora and a dictionary is created along with a
    corresponding index to lookup the words and their keys for the whole
    dataset. All files are saved.

    Inputs
        transcript_paths: str - The location of the transcripts
        current_dir: str - The location of the current working directory to
                     save the time signatures for each file

    Output
        per_file_text: list - Contains a list of lists for every file in the
                       dataset contains a list of sentences for each sentence
                       spoken by the participant
        dict_words: dictionary - The corpora of words and their keys
        dict_words_index: dictionary - The index of words in the dataset
    """
    complete_text_data = []
    per_file_text = []
    special_case = config_process.interrupt
    for i in transcript_paths:
        trial = i.split('/')[-2]
        trial = int(trial.split('_')[0])
        with open(i, 'r') as file:
            data = file.readlines()
        ellies_first_intro = 0
        inter = []
        for j, values in enumerate(data):
            # The headers are in first position
            if j == 0 or len(values) == 1:
                pass
            else:
                # The values in this list are actually strings, not int
                # Create temp which holds onset/offset times and the
                # speaker id
                speaker_id = values.split()[2]
                temp = values.split('\t')[-1][0:-1]
                if len(temp) == 0:
                    time_start = time_end = 0
                else:
                    time_start = float(values.split()[0])
                    time_end = float(values.split()[1])
                if len(values) > 1:
                    sync = values.split()[-1]
                else:
                    sync = ''
                if sync in config_process.synchronise_labels:
                    sync = True
                else:
                    sync = False
                if ellies_first_intro == 0:
                    if speaker_id == 'Ellie' or speaker_id == 'ellie':
                        ellies_first_intro += 1
                    elif sync and trial in config_process.transcripts:
                        ellies_first_intro += 1
                else:

                    if len(temp) > 0 and speaker_id == ('Participant' or
                                                        'participant'):
                        if sync:
                            pass
                        else:
                            if trial in special_case:
                                inter_start = special_case[trial][0]
                                inter_end = special_case[trial][1]
                                if time_start < inter_start < time_end:
                                    pass
                                elif time_start < inter_end < time_end:
                                    pass
                                elif inter_start < time_start < inter_end:
                                    pass
                                elif inter_start < time_end < inter_end:
                                    pass
                                else:
                                    temp = remove_words_symbols(temp)
                                    temp = temp.lower()
                                    if len(temp) == 0:
                                        pass
                                    else:
                                        inter.append(temp)
                                        complete_text_data.append(temp)
                            else:
                                temp = remove_words_symbols(temp)
                                temp = temp.lower()
                                if len(temp) == 0:
                                    pass
                                else:
                                    inter.append(temp)
                                    complete_text_data.append(temp)
        per_file_text.append(inter)

    text_data = [lines.lower().split() for lines in complete_text_data]
    dict_words = corpora.Dictionary(text_data)
    dict_words_index = {values: keys for keys, values in dict_words.items()}
    # nlp = spacy.load('en_core_web_md')
    # # Create an empty dictionary
    # dict_vectors = dict.fromkeys(range(len(dict_words)))
    # for i in range(len(dict_words)):
    #     try:
    #         test_vec = nlp(dict_words[i]).vector
    #         dict_vectors[i] = test_vec
    #     except:
    #         print('an error occured at:', i)
    #         sys.exit(f"Specific word {dict_words[i]} was not found in nlp")
    # save_pickle(dict_vectors, 'dict_vectors', current_dir)
    save_pickle(per_file_text, 'per_file_text', current_dir)
    save_pickle(dict_words, 'dict_words', current_dir)
    save_pickle(dict_words_index, 'dict_words_index', current_dir)

    return per_file_text, dict_words, dict_words_index


def save_pickle(data, name, path=None):
    """
    Saves the data in a .pickle format
    Inputs
        data: - The data to be saved
        name: str - The name of the data to be saved
        path: str - The location to save the data

    """
    if path:
        name = os.path.join(path, name)

    with open(name + '.pickle', 'wb') as file:
        pickle.dump(data, file)


def load_pickle(path):
    """
    Loads data from a pickle file

    Input
        path: str - The location of the pickle data
    """
    with open(path, 'rb') as file:
        return pickle.load(file)


def remove_words_symbols(string):
    """
    Some of the transcript information has odd words, unknown words,
    and in-place symbols to hide private information. In order to remove
    these, this function was created.
    Input
        string: str - The input string from a transcript

    Output
        string: str - The updated string which has been cleaned
    """
    if isinstance(string, str):
        temp = string.split()
    else:
        temp = string[:]
    marked_for_removal = []
    for split_string in temp:
        if split_string in config_process.words_to_remove.values():
            marked_for_removal.append(split_string)
        else:
            if any(symb in split_string for symb in
                   config_process.symbols_to_remove):
                marked_for_removal.append(split_string)
    for pointer in range(len(marked_for_removal)):
        string = string.replace(marked_for_removal[pointer], '')

    if string in config_process.words_to_remove.values():
        string = string.replace(string, '')

    return string


def create_directories(location, folders_to_make):
    """
    Creates a directory (and potential sub directories) at a location

    :Input
        location: location of the new directories
        folders_to_make: List of the sub directories
    """
    if folders_to_make:
        for i in folders_to_make:
            os.mkdir(os.path.join(location, i))


def get_labels_from_dataframe(path):
    """
    Reads database labels from csv file using pandas.

    Input
        path: The location of the database labels csv file

    Output:
        output: List containing the Participant IDs and the classes/scores
    """
    df = pd.read_csv(path)
    output = [df['Participant_ID'].values.tolist(),
              df['PHQ8_Binary'].values.tolist(),
              df['PHQ8_Score'].values.tolist(),
              df['Gender'].values.tolist()]

    return output


def merge_csv(path1, path2, filename):
    """
    Takes 2 paths to 2 .csv files and merges them into one single dataframe.
    This is then saved at the location specified in filename
    :param path1: Path to the first .csv file
    :param path2: Path to the second .csv file
    :param filename: Path to save the concatenated dataframe
    :return:
    """
    a, b, columnsa, columnsb = meta_data_checker(path1, path2)

    columnsb = list(b)
    # This checks that the column headers are the same in the two CSV files
    for i in range(len(columnsa)):
        # If headers are different, re-name
        if columnsa[i] != columnsb[i]:
            b = b.rename(columns={columnsb[i]: columnsa[i]})

    # Create a single dataframe from the 2 CSV files and save
    dataframes = [a, b]
    c = pd.concat(dataframes)
    c = c.sort_values(by=['Participant_ID'])

    c.to_csv(filename, index=False)


def get_dimensions(feature, summary, dim, audio_mode_is_concat_not_shorten):
    """
    Gets the dimensions of some data that has been segmented into specified
    lengths. For instance, if the input is a [100, 200] array and the
    dimension to segment is 50 the result will be [4, 100, 50]

    Inputs
        feature: numpy.array - The data to be used in calculation
        summary: list - Holds meta data about the dataset such as the
                 shortest file, longest file and all file lengths.
        dim: int - The dimensions that the data will be segmented to
        audio_mode_is_concat_not_shorten: bool - Set False if every file is
                                          to be shortened to the shortest
                                          length file in order to deal with
                                          the variable length issue

    Output
        num_extra_dim: int - The number of extra dimensions after segmenting
                       the data
    """
    if audio_mode_is_concat_not_shorten:
        samples_per_feat = summary[1][summary[0].index('ListOfSamples')]
        num_extra_dim = 0
        for i in samples_per_feat:
            current_dimension = (i.shape[1] // dim)
            num_extra_dim += current_dimension
            if not float(num_extra_dim).is_integer():
                num_extra_dim += 1
    else:
        num_extra_dim = (feature.shape[1] // dim)
        if not float(num_extra_dim).is_integer():
            num_extra_dim += 1
        num_extra_dim = num_extra_dim * feature.shape[0]

    return num_extra_dim


def seconds_to_sample(seconds, window_size, overlap=0, hop_length=0,
                      sample_rate=16000, feature_type='logmel'):
    """
    Converts number of seconds into the equivalent number of samples taking
    into account the type of feature. For example raw audio will simply be
    the seconds * sample rate whereas logmel will require further calculation
    as the process of creating logmel compresses the data along the time axis

    Inputs:
        seconds: Number of seconds to convert
        window_size: Length of window used in feature extraction of logmel
                     for example
        overlap: Overlap used in feature extraction for logmel for example
        hop_length: Hop length used in feature extraction of logmel for example
        sample_rate: Original sampling rate of the data
        feature_type: What type of feature is used? Raw audio? Logmel?

    Outputs:
        samples: Converted samples
    """
    if overlap == 0 and hop_length == 0:
        hop_length = window_size // 2
    elif hop_length == 0 and overlap != 0:
        overlap = overlap / 100
        overlap = window_size * overlap
        hop_length = window_size - round(overlap)

    num_sample = seconds * sample_rate
    if feature_type == 'raw':
        samples = int(num_sample)
    else:
        num_sample = num_sample - (window_size/2)
        num_sample = num_sample // hop_length
        samples = int(num_sample + 2)

    return samples


def count_classes(complete_classes):
    """
    Counts the number of zeros and ones in the dataset:

    Input:
        complete_classes: List of the classes of the dataset

    Outputs:
        zeros: Number of zeros in the dataset
        index_zeros: Indexes of the zeros in the dataset w.r.t. feature array
        ones: Number of ones in the dataset
        index_ones: Indexes of the ones in the dataset w.r.t. feature array
    """
    zeros = 0
    ones = 0
    index_zeros = []
    index_ones = []
    for i, d in enumerate(complete_classes):
        if d == 0:
            zeros += 1
            index_zeros.append(i)
        else:
            ones += 1
            index_ones.append(i)

    return zeros, index_zeros, ones, index_ones


def load_data(path, labels):
    """
    Loads specific data from a dataset using indexes from labels.

    Input:
        path: The location to the database
        labels: The database labels which include the indexes of the specific
                data to load

    Output:
        features: The dataset features
    """
    with h5py.File(path, 'r') as h5:
        features = h5['features'][:]

    features = features[labels[-1].tolist()]

    return features


def load_labels(path):
    """
    Loads the labels for a dataset at a given location.

    Input:
        path: The location to the database labels

    Output:
        labels: The labels for the dataset
    """
    if isinstance(path, list):
        for i, file in enumerate(path):
            with open(file, 'rb') as f:
                if i == 0:
                    labels = pickle.load(f)
                else:
                    labels = np.concatenate((labels,
                                             pickle.load(f)),
                                            axis=1)
    else:
        with open(path, 'rb') as f:
            labels = pickle.load(f)

    return labels


def str2bool(arg_value):
    """
    When parsing in boolean values, for some reason argparse doesn't register
    the initial values, therefore it will always output as True, even if they
    are parsed in as False. This function is used in place of the type
    argument in the argparse.add_argument and fixes this issue. From
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with
    -argparse

    Input
        arg_value: Value parsed in as an argument

    """

    if isinstance(arg_value, bool):
        return arg_value
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def setup_logger(current_directory):
    """
    Setup the logger for the current experiment

    Input
        current_directory: The location of the logger to be stored

    Output
        main_logger: The logger to be used throughout the experiment
    """
    log_path = os.path.join(current_directory, 'audio_file_analysis.log')
    main_logger = logging.getLogger('MainLogger')
    main_logger.setLevel(logging.INFO)
    main_handler = logging.handlers.RotatingFileHandler(log_path)
    main_logger.addHandler(main_handler)

    return main_logger


def csv_read(file, start=None, end=None):
    """
    Read a csv (comma separated value) file and append each line to a list

    Input:
        file: The location of the csv file
        start: Start location for a read line
        end: End location for a read line

    Output:
        data: List of each row from csv file
    """
    data = []
    with open(file) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                if start is not None and end is not None:
                    data.append(row[start:end])
                else:
                    data.append(row)
    return data


def remove_directory(location):
    """
    Removes a directory and all sub directories at a specific location

    Input:
        location: Location of the directory to be removed
    """
    shutil.rmtree(location, ignore_errors=False, onerror=None)


def normalise(data, mean, std):
    """
    From a set of data, normalise the data using the mean and the standard
    deviation to obtain 0 mean and standard deviation of 1

    Inputs:
        data: The data to be processed
        mean: The mean of the data
        std: The standard deviation of the data

    Output:
        normalised_data: Output normalised data with mean 0 and standard
                         deviation of 1
    """
    normalised_data = (data-mean) / std

    return normalised_data
