# Particle field simulation

Simulating the particle fields by taking the time-evolution equations extracted from Euler-Lagrange equations of fields.

This program requires Nvidia GPU (for CUDA.jl)


## Simple example: scalar field

As the Lagrangian density (and its Euler-Lagrange equation / the Klein-Gordon equation) can be constructed from the mass and spring system, I am now doing the reverse:

$$
0 = \partial_\mu \partial^\mu \phi + m^2 \phi \\
= \partial_t^2 \phi - \nabla^2 \phi + m^2 \phi \\
\Rightarrow \partial_t^2 \phi = \nabla^2 \phi - m^2 \phi
$$

This is a Newtonian mechanic time evolution equation. To compute it, I will reserve the memory space for:

- $` \phi `$
- $` \partial_t \phi `$

at each point in space.  

I now set pseudo-infinitesimal steps in space-time: $` dt `$ and $` dx `$, where $` dt `$ corresponds to the time step of each timestamp and $` dx `$ corresponds to the distance between one site in the grid to the next.

I can then compute $` \nabla^2 \phi `$ at a point by taking the differences with the neighboring sites:

$$
\nabla^2 \phi_{(x_1, x_2, x_3)} \approx (\sum_{j=1}^{N=3} \phi_{x_j - 1} + \phi_{x_j + 1} - 2 \phi_{x_j})/dx^2 \\
= [\left(\sum_{\text{neighbor}} \phi\right) - 2N \phi] / dx^2
$$

In each increment of time, I add $` (\partial_t^2 \phi \cdot dt) `$ to $` \partial_t \phi `$ and $` (\partial_t \phi \cdot dt) `$ to $` \phi `$.

https://github.com/user-attachments/assets/f3ae8119-a83e-4cc9-b793-4096d1a293bb

---

## Electron and photon field

Expanding the electron-photon field Lagrangian density:

$$
\mathcal{L} = i \bar{\psi} \gamma^\mu \partial_\mu \psi - m \bar{\psi} \psi - J^\mu A_\mu - \frac{1}{4} F_{\mu\nu} F^{\mu\nu} \\
$$
$$
= i \bar{\psi} \gamma^\mu \partial_\mu \psi - m \bar{\psi} \psi - q \bar{\psi} \gamma^\mu \psi A_\mu \\ - \frac{1}{4} (\partial_\mu A_\nu - \partial_\nu A_\mu)(\partial^\mu A^\nu - \partial^\nu A^\mu)
$$


## Time evolution of $` \psi `$

Deriving the Euler-Lagrange equation for $` \psi `$:

$$
0 = \partial_\mu \left( \frac{\partial \mathcal{L}}{\partial(\partial_\mu \bar{\psi})} \right) - \frac{\partial \mathcal{L}}{\partial \bar{\psi}} \\
= 0 - (i \gamma^\mu \partial_\mu \psi - m \psi - q \gamma^\mu \psi A_\mu) \\
$$
$$
\Rightarrow i \gamma^t \partial_t \psi = m \psi + q \gamma^\mu \psi A_\mu - i \vec{\gamma} \cdot \vec{\nabla} \psi \\
$$

(multiply with $` -i \gamma^t`$)

$$
\Rightarrow \partial_t \psi = -i m \gamma^t \psi - i q \gamma^t \gamma^\mu \psi A_\mu - \gamma^t \vec{\gamma} \cdot \vec{\nabla} \psi
$$

As shown above, the field data I need to store is the $` \psi `$ value for each point in space. In each timestamp, I will add the value $` (\partial_t \psi \cdot dt) `$ to $` \psi `$, where $` dt `$ is the pseudo-infinitesimal timescale.

---

## Time evolution of $` A^\mu `$

Next, doing the same for $` A^\mu `$ becomes a bit tricky. The Euler-Lagrange equation for $` A^\mu `$:

$$
J^\nu = \partial_\mu F^{\mu\nu} = \partial_\mu \partial^\mu A^\nu - \partial^\nu (\partial_\mu A^\mu) \\
$$
$$
= q \bar{\psi} \gamma^\nu \psi \\
$$
$$
= \partial_t^2 A^\nu - \nabla^2 A^\nu - \partial^\nu (\partial_t A^t + \nabla \cdot \vec{A})
$$

Now considering the individual cases of $` \nu `$:

### Case $` \nu = t `$ :

$$
q \bar{\psi} \gamma^t \psi = q \psi^\dagger \psi 
$$

$$
= \cancel{\partial_t^2 A^t} - \nabla^2 A^t - \partial^t (\cancel{\partial_t A^t} + \nabla \cdot \vec{A})
$$

```math
= - \underbrace{\nabla^2 A^t}_{P} - \underbrace{\partial_t (\nabla \cdot \vec{A})}_{Q}
```

### Case $` \nu = x `$ :

$$
q \bar{\psi} \gamma^x \psi 
$$

```math
= \partial_t^2 A^x - \underbrace{\nabla^2 A^x}_{R^x} + \underbrace{\partial_x (\partial_t A^t + \nabla \cdot \vec{A})}_{S^x}
```

This looks difficult to implement!!!

---

### Strategy to implement $` A^\mu `$ time evolution

As shown above, I have named each term in the equations with P, Q, R, and S.

To implement, I will reserve memory for $` A^\mu `$ and $` \partial_\mu A^\nu `$ (a 4x4 Jacobian matrix) at each grid point.

The challenge is how to get the time evolution of $` A^t `$.

My strategy is to compute the values of $`A^t`$ directly and then update $`partial_t A^t`$ in the matrix, which is the opposite of what I do with other field components.

We see that we are able to compute $` Q = \partial_t (\nabla \cdot \vec{A}) = \sum_j \partial_j (\partial_t A^j) `$ using the values in the $` \partial_\mu A^\nu `$ matrix.  
Then we can obtain the value of $` P `$.

$$
P(i, k, l) = \nabla^2 A^t = \sum_j \partial_j^2 A^t \\
$$
$$
\approx \sum_{j=1}^{3} \frac{A^t_{i-1} + A^t_{i+1} - 2 A^t_i}{dx^2} \\
$$
$$
= \frac{\left( \sum_{\text{all neighbors}} A^t_i \right) - 2N A^t}{dx^2} \\
$$
$$
\Rightarrow A^t = \frac{1}{2N} \left( \sum_{\text{all neighbors}} A^t_i - P dx^2 \right) \\
$$
$$
\quad (N = 3)
$$

With this, I can now get the updated $` A^t `$, and calculate and update the $`\partial_t A^t = \frac{A^t_{\text{new}} - A^t_{\text{old}}}{dt}`$ in the matrix.

We see that the above uses $`(partial_t A^j)`$; we need to update them at each timestamp.

We can do so by obtaining the acceleration $`(\partial_t^2 A^x)`$ by calculating $R^j$ and $S^j$. It is feasible by using the $`A^\mu`$ values at the point and its neighbor (For $`S^j`$, I also need to use $`\partial_t A^t`$  that we just calculated above.)
We will add ($`\partial_t^2 A^j dt`$) to ($`\partial_t A^j`$) in the matrix, and we can finally update the spacial components $`A^j`$.

