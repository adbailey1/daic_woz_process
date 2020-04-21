from unit_test.unit_test_audio_file_analysis import run_tests_afa
from unit_test.unit_test_audio_feature_extractor import run_tests_afe
import os
import numpy as np
import random
import time
from unit_test import config_test


def main():
    current_path = os.path.dirname(os.path.realpath(__file__))
    np.random.seed(1234)
    random.seed(1234)
    total_time_start = time.time()
    run_tests_afa(config_test, current_path)
    # run_tests_afe(config_test)
    total_time_end = time.time()
    print('Unit Tests Took: ', total_time_end-total_time_start, 's')


if __name__ == "__main__":
    main()
