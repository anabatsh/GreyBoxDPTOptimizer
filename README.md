### Decision Pretrained Transformer

To run DPT one simply needs execute the following command:

````bash
python run.py
````

---

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
```

