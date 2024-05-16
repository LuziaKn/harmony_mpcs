# Harmony MPCs
<p align="center">
    <img src="./assets/example.png"  alt="1" height = 200px >
</p>

Repo including a simple mpc planner for a mobile base. The implementation is completely independent of the Robot Operating System (ROS) but can be combined with ROS1 (and soon ROS2) via [harmony_mpcs_ros](https://github.com/LuziaKn/harmony_mpcs_ros).

## Requirements
The implementation currently uses **Embotech's ForcesPro** solver. The license is free for educational use. 

You have to request a license for [forcespro](https://forces.embotech.com/) and install it
according to their [documentation](https://forces.embotech.com/Documentation/installation/obtaining.html#sec-obtaining).
Helpful information about assigning the license to your computer can be found [here](https://my.embotech.com/manual/system_information).

To make use of the ForcesPro solver with python follow their instructions [here](https://forces.embotech.com/Documentation/installation/python.html#python).

The location of the python package `forcespro` must also be included in your python path.

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/forces/pro"
```
Consider adding it to your `.bashrc` (`.zshrc`) file

## Installation using poetry
We use poetry to install python dependencies in a virtual environment.
Install poetry by following the instructions provided in the [poetry-docs](https://python-poetry.org/docs/).

Then run

```bash
python install
```
and

Source the generated environment using 

```bash
python shell
```
 and install the harmony_mpcs package via 
 
```bash
pip install -e .
```


## Examples
 Genterate the solver:

 ```bash
 cd harmony_mpcs/harmony_mpcs/examples 
 python generate_solver.py
 ```
Run the example

```bash
python dingo_urdfenvs_example.py
```

The parameters of the mpc can be adapted in the config file in examples/config.
