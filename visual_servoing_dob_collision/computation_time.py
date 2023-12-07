import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import os
from utils.dict_utils import load_dict



if __name__ == '__main__':
    # exp_names = ['exp_005_1', 'exp_005_2', 'exp_005_3', 'exp_005_4', 'exp_005_5']
    exp_names = ['exp_006_1', 'exp_006_2', 'exp_006_3', 'exp_006_4', 'exp_006_5']
    # exp_names = ['exp_007_1', 'exp_007_2', 'exp_007_3', 'exp_007_4', 'exp_007_5']
    # exp_names = ['exp_008_1', 'exp_008_2', 'exp_008_3', 'exp_008_4', 'exp_008_5']

    for i in range(len(exp_names)):
        exp_name = exp_names[i]
        results_dir = "{}/results_collision_dob/".format(str(Path(__file__).parent.parent)) + exp_name
        summary = load_dict("{}/summary.npy".format(results_dir))
        cbf_computation_time = summary['cbf_computation_time']
        cvxpylayer_computation_time = summary['cvxpylayer_computation_time']
        stop_ind = summary["stop_ind"]

        if i == 0:
            total_cbf_computation_time = cbf_computation_time[:stop_ind]
            total_cvxpylayer_computation_time = cvxpylayer_computation_time[:stop_ind]
        else:
            total_cbf_computation_time = np.concatenate((total_cbf_computation_time, cbf_computation_time[:stop_ind]))
            total_cvxpylayer_computation_time = np.concatenate((total_cvxpylayer_computation_time, cvxpylayer_computation_time[:stop_ind]))

    # Print mean and std of computation time
    print("Total data length: {}".format(len(total_cbf_computation_time)))
    print("Mean CBF computation time: {}".format(np.mean(total_cbf_computation_time)))
    print("Std CBF computation time: {}".format(np.std(total_cbf_computation_time)))

    print("Total data length: {}".format(len(total_cvxpylayer_computation_time)))
    print("Mean cvxpylayer computation time: {}".format(np.mean(total_cvxpylayer_computation_time)))
    print("Std cvxpylayer computation time: {}".format(np.std(total_cvxpylayer_computation_time)))