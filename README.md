### Decision Pretrained Transformer for Black Box Optimization

````bash
python create_problems.py
python run_solvers.py
python run.py
````

Run `create_problems` to create a train, val and test sets for various QUBO problems. Then, run `run_solvers` to run all solvers on the test set. Finally, run `run` to choose the best optimizer and run it in the train and val sets.
Thus, a value `info` in every problem (from train, val and test sets) would be defined. 

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

