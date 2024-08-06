import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial
import os
import json
from time import perf_counter as tpc
from .utils import *


def _generate_initial(d, n, r, key):
    """Build initial random TT-tensor for probability."""
    keyl, keym, keyr = jax.random.split(key, 3)

    Yl = jax.random.uniform(keyl, (1, n, r))
    Ym = jax.random.uniform(keym, (d-2, r, n, r))
    Yr = jax.random.uniform(keyr, (r, n, 1))

    return [Yl, Ym, Yr]

def _interface_matrices(Ym, Yr):
    """Compute the "interface matrices" for the TT-tensor."""
    def body(Z, Y_cur):
        Z = jnp.sum(Y_cur, axis=1) @ Z
        Z /= jnp.linalg.norm(Z)
        return Z, Z

    Z, Zr = body(jnp.ones(1), Yr)
    _, Zm = jax.lax.scan(body, Z, Ym, reverse=True)

    return jnp.vstack((Zm, Zr))

def _likelihood(Yl, Ym, Yr, Zm, i):
    """Compute the likelihood in a multi-index i for TT-tensor."""
    def body(Q, data):
        I_cur, Y_cur, Z_cur = data

        G = jnp.einsum('r,riq,q->i', Q, Y_cur, Z_cur)
        G = jnp.abs(G)
        G /= jnp.sum(G)

        Q = jnp.einsum('r,rq->q', Q, Y_cur[:, I_cur, :])
        Q /= jnp.linalg.norm(Q)

        return Q, G[I_cur]

    Q, yl = body(jnp.ones(1), (i[0], Yl, Zm[0]))
    Q, ym = jax.lax.scan(body, Q, (i[1:-1], Ym, Zm[1:]))
    Q, yr = body(Q, (i[-1], Yr, jnp.ones(1)))

    y = jnp.hstack((jnp.array(yl), ym, jnp.array(yr)))
    return jnp.sum(jnp.log(jnp.array(y)))

def _sample(Yl, Ym, Yr, Zm, key):
    """Generate sample according to given probability TT-tensor."""
    def body(Q, data):
        key_cur, Y_cur, Z_cur = data

        G = jnp.einsum('r,riq,q->i', Q, Y_cur, Z_cur)
        G = jnp.abs(G)
        G /= jnp.sum(G)

        i = jax.random.choice(key_cur, jnp.arange(Y_cur.shape[1]), p=G)

        Q = jnp.einsum('r,rq->q', Q, Y_cur[:, i, :])
        Q /= jnp.linalg.norm(Q)

        return Q, i

    keys = jax.random.split(key, len(Ym) + 2)

    Q, il = body(jnp.ones(1), (keys[0], Yl, Zm[0]))
    Q, im = jax.lax.scan(body, Q, (keys[1:-1], Ym, Zm[1:]))
    Q, ir = body(Q, (keys[-1], Yr, jnp.ones(1)))

    il = jnp.array(il, dtype=jnp.int32)
    ir = jnp.array(ir, dtype=jnp.int32)
    return jnp.hstack((il, im, ir))

class PROTES(Solver):
    def __init__(self, problem, budget, k_init=0, k_samples=100, k_top=10):
        super().__init__(problem, budget, k_init=0, k_samples=k_samples)
        self.k_top = k_top

    def init_settings(self, seed=0):
        self.rng = jax.random.PRNGKey(seed)
        self.rng, key = jax.random.split(self.rng)
        self.P = _generate_initial(d=self.problem.d, n=self.problem.n, r=5, key=key)

        optim = optax.adam(5.E-2)
        self.state = optim.init(self.P)

        interface_matrices = jax.jit(_interface_matrices)
        sample = jax.jit(jax.vmap(_sample, (None, None, None, None, 0)))
        likelihood = jax.jit(jax.vmap(_likelihood, (None, None, None, None, 0)))

        @jax.jit
        def loss(P_cur, I_cur):
            Pl, Pm, Pr = P_cur
            Zm = interface_matrices(Pm, Pr)
            l = likelihood(Pl, Pm, Pr, Zm, I_cur)
            return jnp.mean(-l)

        loss_grad = jax.grad(loss)

        @jax.jit
        def optimize(state, P_cur, I_cur):
            grads = loss_grad(P_cur, I_cur)
            updates, state = optim.update(grads, state)
            P_cur = jax.tree_util.tree_map(lambda p, u: p + u, P_cur, updates)
            return state, P_cur
        
        self.jx_optimize = optimize
        self.jx_interface_matrices = interface_matrices
        self.jx_sample = sample

    def sample_points(self):
        Pl, Pm, Pr = self.P
        Zm = self.jx_interface_matrices(Pm, Pr)
        self.rng, key = jax.random.split(self.rng)
        I = self.jx_sample(Pl, Pm, Pr, Zm, jax.random.split(key, self.k_samples))
        return I

    def update(self, points):
        targets = jnp.array(self.problem.target(np.array(points)))
        ind = jnp.argsort(targets)[:self.k_top]
        points = points[ind]
        targets = targets[ind]
        # for _ in range(self.k_gd):
        self.state, self.P = self.jx_optimize(self.state, self.P, points)        
        points, targets = np.array(points), np.array(targets)
        return points, targets

class PROTESBO(Solver):
    def __init__(self, problem, budget, k_init=0, k_samples=100):
        super().__init__(problem, budget, k_init=0, k_samples=k_samples)

    def init_settings(self, seed=0):
        self.rng = jax.random.PRNGKey(seed)
        self.rng, key = jax.random.split(self.rng)
        self.P = _generate_initial(d=self.problem.d, n=self.problem.n, r=5, key=key)

        optim = optax.adam(5.E-2)
        self.state = optim.init(self.P)

        interface_matrices = jax.jit(_interface_matrices)
        sample = jax.jit(jax.vmap(_sample, (None, None, None, None, 0)))
        likelihood = jax.jit(jax.vmap(_likelihood, (None, None, None, None, 0)))

        @jax.jit
        def loss(P_cur, I_cur):
            Pl, Pm, Pr = P_cur
            Zm = interface_matrices(Pm, Pr)
            l = likelihood(Pl, Pm, Pr, Zm, I_cur)
            return jnp.mean(-l)

        loss_grad = jax.grad(loss)

        @jax.jit
        def optimize(state, P_cur, I_cur):
            grads = loss_grad(P_cur, I_cur)
            updates, state = optim.update(grads, state)
            P_cur = jax.tree_util.tree_map(lambda p, u: p + u, P_cur, updates)
            return state, P_cur
        
        self.jx_optimize = optimize
        self.jx_interface_matrices = interface_matrices
        self.jx_sample = sample

    def sample_points(self):
        Pl, Pm, Pr = self.P
        Zm = self.jx_interface_matrices(Pm, Pr)
        self.rng, key = jax.random.split(self.rng)
        I = self.jx_sample(Pl, Pm, Pr, Zm, jax.random.split(key, self.k_samples))
        return I

    def update(self, points):
        targets = jnp.array(self.problem.target(np.array(points)))
        ind = jnp.argsort(targets)[:self.k_top]
        points = points[ind]
        targets = targets[ind]
        # for _ in range(self.k_gd):
        self.state, self.P = self.jx_optimize(self.state, self.P, points)        
        points, targets = np.array(points), np.array(targets)
        return points, targets