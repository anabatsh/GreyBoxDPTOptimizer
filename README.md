### Decision Pretrained Transformer for Black Box Optimization

````bash
python generate_data.py --args
````

Run `generate_data` to create train, val and test sets for various QUBO problems, run solvers on them, save the results and set the `info` variable for every problem in the sets with the found minimums. Alternatively, do this step by step by running `create_problems` to create datasets of a specified problem, `run_solvers` to solve the problems from a specified set with a specified solver and save the results, and, finally, `update_info` to set the `info` variable for every problem in a specified set with the found minimum.

````bash
python create_problems.py --args
python run_solvers.py --args
python run.py --args
````

Then, run `train_dpt` to train a DPT model on the train sets and validate on the val sets. In order to test the model, go to `notebooks/test_model` and run corresponding cells. There a model can be loaded from a particular checkpoint and run on a specified problem set. 

Moreover, `notebooks/test_data` is available to analyze a problem set, namely to obtain some statistics and print additional information. 

<!-- To run DPT one simply needs execute the following command:

````bash
python run.py
```` -->

---
<!-- 
### Grey Box

A repo for solving **Integer Nonlinear Optimization** problems:

$$
\begin{align}
& \text{minimize} \ f(x) \ \text{for} \ x = [\{0, ..., n-1\}]^d \\
\end{align}
$$

Here $f$ is called **target function**, and functions $g_i, h_i$ are referred to as the **constraints**. 

All the problems are presented in the `problems` module as classes, inheriting from a base class `Problem`, located in `problem.base`. 

`Problem` class has two mandatory arguments:

- `d`​ - the dimensionality of the problem
- `n` - the mode of the problem

and a mandatory method:

- `target` - to get the target value for a given argument value

```python
problem = Problem(d=10, n=2)
```

To visualize a problem, one could simply call the `show_problem` method from the `utils` module with an additional argument `save_dir` to specify the path to save the result:

```python
show_problem(problem, save_dir)
``` -->

