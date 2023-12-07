import sys
import os
import json
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def write_header(file):
    data = f"""<?xml version="1.0" ?>
<robot name="box">

    <material name="panda_white">
        <color rgba="1. 0.87 0.68 1."/>
    </material>

    <material name="metal">
        <color rgba="0.6 0.6 0.6 1."/>
    </material>

    <material name="orangered">
        <color rgba="1 0.34 0.2 0.5"/>
    </material>\n"""
    file.write(data)

def write_footer(file):
    data = f"""\n</robot>\n"""
    file.write(data)

def write_box(file, width, length):
    data = f"""\n
    <link name="link1">
        <inertial>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <mass value=".0"/>
            <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
        </inertial>

        <visual>
            <origin rpy="0 0 0" xyz="0.0 0 0.25"/>
            <geometry>
                <box size="{width} {length} 0.005"/>
            </geometry>
            <material name="panda_white"/>
        </visual>
    </link>\n"""
    file.write(data)

def write_legs(file, width, length):
    data = f"""
    <link name="link2">
        <inertial>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <mass value=".0"/>
            <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
        </inertial>\n"""
    
    positions = np.array([[1,1],
                          [1,-1],
                          [-1,1],
                          [-1,-1]], dtype=np.float32) * (width/2-0.01, length/2-0.01)
    for i in range(len(positions)):
        data += f"""
        <visual>
            <origin rpy="0 0 0" xyz="{positions[i,0]} {positions[i,1]} 0.125"/>
            <geometry>
                <cylinder radius="0.005" length="0.25"/>
            </geometry>
            <material name="metal"/>
        </visual>\n"""

    data += f"""
    </link>
    \n"""
    file.write(data)

def write_spheres(file, width, length, sphere_per_unit):
    data = f"""
    <link name="link3">
        <inertial>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <mass value=".0"/>
            <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
        </inertial>\n"""
    
    x = np.arange(0, width, 1/sphere_per_unit) + 1/sphere_per_unit/2 - width/2
    y = np.arange(0, length, 1/sphere_per_unit) + 1/sphere_per_unit/2 - length/2
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()]).T
    positions = np.round(positions,2)
    radius  = 1/sphere_per_unit/2*np.sqrt(2) + 0.001

    for i in range(len(positions)):
        data += f"""
        <visual>
            <origin rpy="0 0 0" xyz="{positions[i,0]} {positions[i,1]} 0.25"/>
            <geometry>
                <sphere radius="0.074"/>
            </geometry>
            <material name="orangered"/>
        </visual>\n"""

    data += f"""
    </link>
    \n"""
    file.write(data)
    return positions

def write_joints(file, sphere=False):
    data = f"""
    <joint name="fixed_joint" type="fixed">
        <parent link="link1"/>
        <child link="link2"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
    </joint>\n"""
    if sphere:
        data += f"""
    <joint name="fixed_joint2" type="fixed">
        <parent link="link1"/>
        <child link="link3"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
    </joint>
    \n"""
    file.write(data)

#################################################
width = 1.0
length = 1.0
sphere_per_unit = 10

filename = "box_big_with_{:d}_spheres.urdf".format(int(width*length*sphere_per_unit**2))
filename = os.path.join(str(Path(__file__).parent), filename)
with open(filename, 'w') as file:
    write_header(file)
    write_box(file, width, length)
    write_legs(file, width, length)
    sphere_positions = write_spheres(file, width, length, sphere_per_unit)
    write_joints(file, sphere=True)
    write_footer(file)

filename = "box_big_wo_{:d}_spheres.urdf".format(int(width*length*sphere_per_unit**2))
filename = os.path.join(str(Path(__file__).parent), filename)
with open(filename, 'w') as file:
    write_header(file)
    write_box(file, width, length)
    write_legs(file, width, length)
    write_joints(file, sphere=False)
    write_footer(file)

# Save sphere positions as json
filename = "box_big_with_{:d}_spheres_pos.json".format(int(width*length*sphere_per_unit**2))
filename = os.path.join(str(Path(__file__).parent), filename)
sphere_positions = np.hstack([sphere_positions, 0.25*np.ones((len(sphere_positions),1))])
data = {"sphere_centers": sphere_positions.tolist()}

json_str = json.dumps(data, separators=(',', ':'))
# Replacing '],[' with '],\n[' to format the inner arrays on new lines
formatted_json_str = json_str.replace('],[', '],\n[')

# Saving the formatted data to a JSON file
with open(filename, 'w') as file:
    file.write(formatted_json_str)
