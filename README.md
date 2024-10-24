### Grey Box

A repo for solving **Integer Nonlinear Optimization** problems:

$$
\begin{align}
& \text{minimize} \ f(x) \ \text{for} \ x = [\{1, ..., n\}]^d \\
& \text{such that} \\
& g_i(x) \leqslant 0 \ \text{for} \ i \in \{1, ..., k_g\} \\
& h_i(x) = 0 \ \text{for} \ i \in \{1, ..., k_h\}
\end{align}
$$

Here $f$ is called **target function**, and functions $g_i, h_i$ are referred to as the **constraints**. 

All the problems are presented in the `problems` module as classes, inheriting from a base class `Problem`, located in `problem.base`. 

`Problem` class has two mandatory arguments:

- `d`​ - the dimensionality of the problem
- `n` - the mode of the problem

and two mandatory methods:

- `target` - to get the target value for a given argument value
- `constraints` - to get the values of all the constraint functions for a given argument value

```python
problem = Net(d=10, n=2)
```

To visualize a problem, one could simply call the `show_problem` method from the `utils` module with an additional argument `save_dir` to specify the path to save the result:

```python
show_problem(problem, save_dir)
```

That creates a file `save_dir/results.png` containing the corresponding picture.

We suggest solving the aforementioned problems with different solvers, leveraging a common **sample-update** strategy:

0. **Warmstart**: generate the init set $D_n = \{(x_i, y_i)\}^n_{i=1}$ ​​
1. **Sampling**: new $k$ points $x_{n+1}, ..., x_{n+k}$ are sampled according to the current $D_n$ 
2. **Substitution**: for the sampled points $x_{n+1}, ..., x_{n+k}$  the corresponding target values are calculated $y_{n+1} = f(x_{n+1}), ..., y_{n+k} = f(x_{n+k})$ 
3. **Updating**: the current set $D_n$ is updated with the selected pairs: $D_{n+1} = \{D_n, (x_{n+1}, y_{n+1})... (x_{n+k}, y_{n+k})\}$
4. Repeat until convergency

<!-- <img src="/Users/anabatsh/Library/Application Support/typora-user-images/Screenshot 2024-10-08 at 18.54.01.png" alt="Screenshot 2024-10-08 at 18.54.01" style="zoom:20%;" /> -->

These solvers are presented in the `solvers` module as classes, inheriting a base class `Solver` located in `solvers.base`.

`Solver` class has four mandatory arguments:

- `problem` - a problem of the described class that is to be solved
- `budget` - maximum allowed number of calls to the target function
- `k_init` - $n$ value from the formula - number of initially generated pairs for the set $D_n$
- `k_samples` - $k$ value from the formula - number of generated points on each step

and three mandatory methods:

- `init_points` - to perform the warmstart step
- `sample_points` - to perform the sampling step

- `update` - to perform the updating step
- `optimize` - to perform whole optimization

Example:

```python
solver = BO(problem, budget=20, k_init=10, k_samples=1)
```

To perform the optimization, you need to call the `optimize` method with an argument `save_dir` to specify the path to save the results:

```python
solver.optimize(save_dir)
```

this method doesn't return anything directly but logs all the information in `Logger`, which is another helpful class from `solvers.base`, which helps to log every step of the optimization process, namely:

- `t_best` - time consumption of the optimization
- `y_best` - the best-found target value
- `x_best` - the argument value corresponding to y_best 
- `m_list` - a list of iterations on which the solver had updated the best-found solution because it found a better one 
- `y_list` - a history of best-found target values per iteration from m_list

So, after performing an optimization method, one can either manually print out the `solver.logger.logs` to get all the information described above or simply check out the file `save_dir/results.json` that the logger creates automatically during the optimization process.

Additionally, the logger creates a file `save_dir/logs.txt` that contains a whole story of the optimization process in the form of a sampled point and its target value per each optimization step.

To visualize all the results, one could simply call the `show_results` function from the `utils` module with an argument `save_dir` to specify the path to read (!) and save all the information:

```python
show_results(save_dir)
```

This creates two files: `save_dir/results.png` and `save_dir/results.txt`. The former contains a picture, depicting entire optimization processes for comparison and the latter directly compares the final results in a form of a table.

A terminal command to run an experiment manually:

```bash
python ./run.py 
--problem Net                  # problem
--d 10                         # dimensionality of the problem
--n 2                          # mode of the problem
--problem_kwargs '{"seed":1}'  # additional parameters of the problem
--solver PROTES                # solver
--budget 10                    # maximum number of calls to the target function
--k_init 0                     # number of initial points for a warmstart
--k_samples 10                 # number of points sampled on each step
--solver_kwargs '{"k_top": 2}' # additional parameters of the solver
--n_runs 2                     # number of reruns for each solver
--save_dir results             # directory to save the results
```

To run several experiments, one can edit the `run.sh` file and execute

```bash
bash run.sh
```

Note that the default `run.sh` file already contains all the benchmarks and calls to visualization functions. One can duplicate it to set another problem or set of hyperparameters.
