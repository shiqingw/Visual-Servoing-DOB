import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import os
from utils.dict_utils import load_dict



if __name__ == '__main__':
    exp_names = ['exp_005']

    for i in range(len(exp_names)):
        exp_name = exp_names[i]
        results_dir = "{}/results_collision_dob/".format(str(Path(__file__).parent.parent)) + exp_name
        summary = load_dict("{}/summary.npy".format(results_dir))
        cbf_computation_time = summary['cbf_computation_time']
        cvxpylayer_computation_time = summary['cvxpylayer_computation_time']
        stop_ind = summary["stop_ind"]

        # Print mean and std of computation time
        print("Mean CBF computation time: {}".format(np.mean(cbf_computation_time[:stop_ind])))
        print("Std CBF computation time: {}".format(np.std(cbf_computation_time[:stop_ind])))

        print("Mean cvxpylayer computation time: {}".format(np.mean(cvxpylayer_computation_time[:stop_ind])))
        print("Std cvxpylayer computation time: {}".format(np.std(cvxpylayer_computation_time[:stop_ind])))