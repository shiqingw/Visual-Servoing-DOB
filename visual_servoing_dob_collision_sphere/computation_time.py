import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import os
from utils.dict_utils import load_dict



if __name__ == '__main__':
    exp_names = ['exp_002']

    for i in range(len(exp_names)):
        exp_name = exp_names[i]
        results_dir = "{}/results_collision_dob_sphere/".format(str(Path(__file__).parent.parent)) + exp_name
        summary = load_dict("{}/summary.npy".format(results_dir))
        cbf_computation_time = summary['cbf_computation_time']
        stop_ind = summary["stop_ind"]

        # Print mean and std of computation time
        print("Mean computation time: {}".format(np.mean(cbf_computation_time[:stop_ind])))
        print("Std computation time: {}".format(np.std(cbf_computation_time[:stop_ind])))