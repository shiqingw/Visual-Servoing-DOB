import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import os

# Python code to generate the specified script and save it to a file

def generate_script(exp_num, file_name):
    with open(file_name, "a") as file:
        for i in range(1, 6):
            file.write(f"python visual_servoing_dob_collision/box_under_dob_normalized.py --exp_num {exp_num}\n")
            file.write(f"mv results_collision_dob/exp_{exp_num:03d} results_collision_dob/exp_{exp_num:03d}_{i}\n\n")

# Generate the script and save it to a specified location
filename = os.path.join(str(Path(__file__).parent.parent), "ours.sh")
exp_nums = [5,6,7,8]
for exp_num in exp_nums:
    generate_script(exp_num, filename)

