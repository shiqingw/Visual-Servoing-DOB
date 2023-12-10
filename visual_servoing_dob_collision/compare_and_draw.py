import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import os
from utils.dict_utils import load_dict



if __name__ == '__main__':
    # exp_names = ['exp_003_w_cbf','exp_003_wo_cbf']
    # exp_names = ['exp_004_w_cbf','exp_004_wo_cbf']
    # exp_names = ['exp_009_close','exp_010_far']
    exp_names = ['exp_011_close','exp_012_far']
    # labels = ['w/ CBF', 'w/o CBF']
    # labels = ['target depth: 0.52 m', 'target depth: 0.72 m']
    labels = ['target depth: 0.60 m', 'target depth: 0.80 m']
    linestyles = ['-', '--']
    label_fs = 35
    tick_fs = 30
    legend_fs = 40
    linewidth = 6

    ###############################
    print("==> Plot cbf ...")
    fig, ax = plt.subplots(figsize=(12,9), dpi=100, frameon=True)
    plt.rcParams.update({"text.usetex": True})

    for i in range(len(exp_names)):
        exp_name = exp_names[i]
        results_dir = "{}/results_collision_dob/".format(str(Path(__file__).parent.parent)) + exp_name
        summary = load_dict("{}/summary.npy".format(results_dir))
        times = summary["times"]
        stop_ind = summary["stop_ind"]
        cbf_value = summary["cbf_value"]
        if "exp_004_wo_cbf" in exp_name:
            stop_ind = 75

        plt.plot(times[:stop_ind], cbf_value[:stop_ind], linestyle = linestyles[i],
                linewidth = linewidth, label="$h$" + " ("+labels[i]+")")


    plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted', linewidth = linewidth)
    plt.legend(fontsize = legend_fs)
    plt.xlabel('Time (s)', fontsize=label_fs)
    plt.xlim([np.min(times)-0.5, np.max(times)+0.5])
    plt.ylabel('$h$ values', fontsize=label_fs)
    plt.xticks(fontsize = tick_fs)
    plt.yticks(fontsize = tick_fs)
    plt.grid()
    plt.tight_layout()
    plt.draw()
    results_dir = "{}/results_collision_dob/exp_{:03d}".format(str(Path(__file__).parent.parent), 0)
    name = " ".join(str(x) for x in exp_names)
    name = name.replace(" ", "_")
    plt.savefig(os.path.join(results_dir, name + '.pdf'))
    plt.close(fig)