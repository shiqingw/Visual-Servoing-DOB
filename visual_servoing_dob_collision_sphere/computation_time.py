import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import os
from utils.dict_utils import load_dict



if __name__ == '__main__':
    # exp_names = ['exp_003_1', 'exp_003_2', 'exp_003_3', 'exp_003_4', 'exp_003_5']
    # exp_names = ['exp_004_1', 'exp_004_2', 'exp_004_3', 'exp_004_4', 'exp_004_5']
    # exp_names = ['exp_005_1', 'exp_005_2', 'exp_005_3', 'exp_005_4', 'exp_005_5']
    exp_names = ['exp_006_1', 'exp_006_2', 'exp_006_3', 'exp_006_4', 'exp_006_5']

    for i in range(len(exp_names)):
        exp_name = exp_names[i]
        results_dir = "{}/results_collision_dob_sphere/".format(str(Path(__file__).parent.parent)) + exp_name
        summary = load_dict("{}/summary.npy".format(results_dir))
        cbf_computation_time = summary['cbf_computation_time']
        stop_ind = summary["stop_ind"]

        if i == 0:
            total_cbf_computation_time = cbf_computation_time[:stop_ind]
        else:
            total_cbf_computation_time = np.concatenate((total_cbf_computation_time, cbf_computation_time[:stop_ind]))


    # Print mean and std of computation time
    print("Total data length: {}".format(len(total_cbf_computation_time)))
    print("Mean computation time: {}".format(np.mean(total_cbf_computation_time[:stop_ind])))
    print("Std computation time: {}".format(np.std(total_cbf_computation_time[:stop_ind])))