from config_files import config
import os
from audio import audio_file_analysis
import shutil
import time
from utils import utilities
import pickle
import csv
import random
import numpy as np


def mod_audio_test(current_path):
    """
    This test works from the previously create onset/offset times to remove
    the segments relating to Ellie from the audio track
    :return:
    """
    with open(os.path.join(current_path, 'on_off_times_unit_test.pickle'), \
            'rb') as f:
        on_off_times = pickle.load(f)

    audio_data = np.random.rand(10000000)
    sample_rate = 16000
    sum = 0
    test_values = np.array(on_off_times[0], float)
    test_values = test_values * sample_rate
    test_values = np.array(test_values, int)
    # test_values[0, 1] = audio_data.shape[0]
    for i in test_values:
        sum += (i[1] - i[0])
    # sum = sum * sample_rate
    # size_leftover = audio_data.shape[0] - sum
    mod_audio = audio_file_analysis.modify_audio_file(audio_data,
                                                      on_off_times[0],
                                                      sample_rate)
    os.remove(os.path.join(current_path, 'on_off_times_unit_test.pickle'))
    if sum == mod_audio.shape[0]:
        return True
    else:
        return False


def transcript_processing_test(current_path):
    """
    Test to check the processing of the transcripts and to eliminate Ellie's
    sections in the audio files to leave an audio track comprising solely of
    the participants responses. This specific section is to test the specific
    cases where there is an interruption in the interview or the transcript
    does not contain Ellies sentences and the generic cases as well.
    :return:
    """
    folders = ['300_P', '373_P', '451_P']
    files = ['300_TRANSCRIPT.csv', '373_TRANSCRIPT.csv', '451_TRANSCRIPT.csv']
    transcript_files = []
    special_case = [[373, [395, 481]], [451]]
    ground_truth_special_case_1 = [[0, 60], [120, 180], [240, 300], [360, 395],
                                   [481, 540]]
    participant = ['Ellie', 'Participant']
    for pointer, i in enumerate(folders):
        temp = os.path.join(current_path, i)
        if not os.path.exists(temp):
            os.mkdir(temp)
        transcript_files.append(os.path.join(temp, files[pointer]))
        starter = line_count = 0
        speaker = flipper = 1
        difference = 60
        to_write_to_file = []
        parti = []
        on_off_times = []
        while starter < 501:
            if line_count == 0:
                to_write_to_file.append(['start_time', 'stop_time',
                                         'speaker', 'value'])
            else:
                if speaker == 0 and int(i.split('_')[0]) == 451:
                    pass
                else:
                    if i == '451_P' and line_count == 1:
                        to_write_to_file.append([starter, starter + difference,
                                                 'Participant', '[sync]'])
                    else:
                        to_write_to_file.append([starter, starter+difference,
                                            participant[speaker], '...'])
                # if line_count == 1:
                #     Ellies.append([starter, starter+difference])
                if speaker == 1:
                    parti.append([starter, starter+difference])
                starter = starter + difference + 0.01
                flipper = flipper * -1
                speaker = speaker + flipper

            line_count += 1

        with open(transcript_files[pointer], mode='w') as tfile:
            twriter = csv.writer(tfile, delimiter='\t')
            twriter.writerows(to_write_to_file)

        if i == '451_P':
            print('wait')
        on_off_times_test = utilities.transcript_file_processing([
            transcript_files[pointer]], current_path)

        if int(i.split('_')[0]) == 373:
            parti = ground_truth_special_case_1
        if int(i.split('_')[0]) == 451:
            parti = parti[1:]
            # Ellies[0] = [0, 100]
        # if to_write_to_file[-1][-2] == 'Participant':
        #     Ellies.append([starter, starter+0.01])

        # Ellies[-1][-1] = Ellies[-1][-1] + 1e3
        # parti.reverse()
        correct = error = 0
        for j in list(range(len(parti))):
            temp_ground_truth = parti[j]
            temp_test = on_off_times_test[0][j]
            temp_ground_truth = np.array(temp_ground_truth, int)
            temp_test = np.array(np.round(np.array(temp_test, float)), int)
            if temp_ground_truth[0] == temp_test[0] and temp_ground_truth[1]\
                    == temp_test[1]:
                correct += 1
            else:
                error += 1
        on_off_times.append(on_off_times_test[0])
        # o = utilities.transcript_file_processing([
        #     '/vol/vssp/datasets/singlevideo01/DAIC-WOZ/492_P/492_TRANSCRIPT.csv'
        #     ''], current_path)
        utilities.remove_directory(temp)
        if error > 0:
            print('Errors found in folder: ', int(i.split('_')[0]))
            return False
    with open(os.path.join(current_path, 'on_off_times_unit_test.pickle'),
              'wb') as f:
        pickle.dump(on_off_times, f)
    os.remove(os.path.join(current_path, 'on_off_times.pickle'))

    return True


def database_class_folder_check(folders, classes, config, reduce=False):
    """
    Test to check that the folders and classes read in from a 'test' file are
    found to be correct with respect to the original ground truth data
    :param folders: List of test folders
    :param classes: List of test classes linked to the test folders
    :param reduce:
    :return: Test Passed - True/False
    """
    db = os.path.join(config.DATASET, 'complete_Depression_AVEC2017.csv')
    d = utilities.csv_read(db, 0, 2)

    dnp = np.zeros((189, 2), dtype=int)
    for pointer, i in enumerate(d):
        dnp[pointer] = int(i[0]), int(i[1])

    if reduce:
        temp_folders = []
        temp_classes = []
        for pointer, i in enumerate(folders):
            if i not in temp_folders:
                temp_folders.append(i)
                temp_classes.append(classes[pointer])
        folders = temp_folders
        classes = temp_classes

    correct = error = 0
    for pointer, i in enumerate(folders):
        temp = int(np.where(dnp == int(i))[0])
        if dnp[temp, 1] == int(classes[pointer]):
            correct += 1
        else:
            error += 1

    total_length_folders = len(folders)
    if total_length_folders == correct:
        return True
    else:
        return False


def check_order(folder_list, file_num):
    print(os.path.dirname(os.path.realpath(__file__)))
    if type(folder_list) is not list:
        folder_list = [folder_list]
    if '/' in folder_list[0]:
        is_input_path = True
    else:
        is_input_path = False

    error_counter = 0
    for i in folder_list:
        if is_input_path:
            test_value = i.split('/')[-1]
            test_value = int(test_value[0:3])
        else:
            test_value = int(i.split('_')[0])
        if file_num != test_value:
            error_counter += 1
            print(f"Error, expected: {file_num} but received: {test_value}")

        file_num += 1
        if file_num in config.EXCLUDED_SESSIONS:
            file_num += 1

    if error_counter != 0:
        print('Errors Found - Test Failed')
        return False
    else:
        return True


def check_folder_data_order():
    """
    Test to check that unordered files will be read into the next section of
    processing in the correct order
    :return: True if test is passed
    """
    temp_dir = os.path.dirname(os.path.realpath(__file__))
    temp_loc = os.path.join(temp_dir, 'temp_dir')
    os.mkdir(temp_loc)
    temp_folder = ['300_P', '301_P', '302_P', '303_P']
    temp_wav = ['300_AUDIO.wav', '301_AUDIO.wav', '302_AUDIO.wav',
                '303_AUDIO.wav']
    temp_csv = ['300_TRANSCRIPT.csv', '301_TRANSCRIPT.csv',
                '302_TRANSCRIPT.csv', '303_TRANSCRIPT.csv']
    random.seed(1234)
    indexes = list(range(len(temp_folder)))
    random.shuffle(indexes)
    temp_folder = [temp_folder[i] for i in indexes]
    temp_wav = [temp_wav[i] for i in indexes]
    temp_csv = [temp_csv[i] for i in indexes]

    if type(temp_folder) is not list:
        temp_folder = [temp_folder]
    if type(temp_wav) is not list:
        temp_wav = [temp_wav]
    if type(temp_csv) is not list:
        temp_csv = [temp_csv]
    for pointer, i in enumerate(temp_folder):
        os.mkdir(os.path.join(temp_loc, i))
        os.mknod(os.path.join(temp_loc, i, temp_wav[pointer]))
        os.mknod(os.path.join(temp_loc, i, temp_csv[pointer]))

    folder, audio, trans = audio_file_analysis.get_meta_data(temp_loc)
    file_num = 300

    test_passed = check_order(folder, file_num)
    if test_passed:
        test_passed_2 = check_order(audio, file_num)
        if test_passed_2:
            test_passed_3 = check_order(trans, file_num)

    shutil.rmtree(temp_loc,
                  ignore_errors=False,
                  onerror=None)

    return test_passed_3


def run_tests_afa(config, current_path):
    start = time.time()
    passed = check_folder_data_order()
    end = time.time()
    if passed:
        print(f"Test check_folder_data_order Passed in {end - start}s")
    else:
        print(f"Test check_folder_data_order Failed in {end - start}s")

    start = time.time()
    folder_class_file = os.path.join(current_path, 'test_folder_classes.csv')
    data = utilities.csv_read(folder_class_file)
    folder = [i[0] for i in data]
    classes = [i[1] for i in data]
    passed = database_class_folder_check(folder, classes, config)
    end = time.time()
    if passed:
        print(f"Test database_class_folder_check Passed in {end - start}s")
    else:
        print(f"Test database_class_folder_check Failed in {end - start}s")

    start = time.time()
    test_passed = transcript_processing_test(current_path)
    end = time.time()
    if test_passed:
        print(f"Test transcript_processing_test Passed in {end - start}s")
    else:
        print(f"Test transcript_processing_test Failed in {end - start}s")

    start = time.time()
    passed = mod_audio_test(current_path)
    end = time.time()
    if passed:
        print(f"Test mod_audio_test Passed in {end - start}s")
    else:
        print(f"Test mod_audio_test Failed in {end - start}s")
