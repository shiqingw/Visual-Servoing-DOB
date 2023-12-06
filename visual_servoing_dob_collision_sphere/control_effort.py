import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import os
from utils.dict_utils import load_dict



if __name__ == '__main__':
    exp_names = ['exp_001_wo_spheres']
    # exp_names = ['exp_002_wo_spheres']

    for i in range(len(exp_names)):
        exp_name = exp_names[i]
        results_dir = "{}/results_collision_dob_sphere/".format(str(Path(__file__).parent.parent)) + exp_name
        summary = load_dict("{}/summary.npy".format(results_dir))
        joint_vels = summary['joint_vels']
        stop_ind = summary["stop_ind"]
        times = summary["times"]

        total_control_effort = 0
        for i in range(stop_ind):
            if i != len(times) -1:
                dt = times[i+1] - times[i]
            total_control_effort += joint_vels[i,:] @ joint_vels[i,:]*dt

    print("Total control effort: {}".format(total_control_effort))