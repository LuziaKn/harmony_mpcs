# Harmony MPCs
Repo including a simple mpc planner for a mobile base.

## Installation
We use poetry to install python dependencies in a virtual environment.
Install poetry by following the instructions provided in the [poetry-docs](https://python-poetry.org/docs/).

Then run

```bash
python install
```
and

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
python dingo_example.py
```

The parameters of the mpc can be adapted in the config file in examples/config.
