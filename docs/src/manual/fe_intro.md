# Introduction to FEM

Here we will present a very brief introduction to partial differential equations (PDEs) and
to the finite element method (FEM). Perhaps the simplest PDE of all is the (linear) heat
equation, also known as the Laplace equation. We will use this equation as a demonstrative
example of the method, and demonstrate how we go from the strong format of the equation, to
the weak form, and then finally to the discrete FE problem.

## Strong format

The strong format of the heat equation may be written as:

```math
- \mathbf{\nabla} \cdot \mathbf{q}(u) = b \quad x \in \Omega,
```

where $u$ is the unknown temperature field, $\mathbf{q}$ is the heat flux and $b$ is an
internal heat source. To complete the system of equations we need boundary conditions.
There are different types of boundary conditions, but the most common ones are Dirichlet
-- which means that the solution $u$ is known at some part of the boundary, and Neumann
-- which means that the gradient of the solution, $\mathbf{\nabla}$ is known. For example

```math
u = u^\mathrm{p} \quad \forall \mathbf{x} \in \Gamma_\mathrm{D},\\
\mathbf{q} \cdot \mathbf{n} = q^\mathrm{p} \quad \forall \mathbf{x} \in \Gamma_\mathrm{N},
```

i.e. the temperature is presribed to $u^\mathrm{p}$ at the Dirichlet part of the boundary,
$\Gamma_\mathrm{D}$, and the heat flux is prescribed to $q^\mathrm{p}$ at the Neumann part
of the boundary, $\Gamma_\mathrm{N}$.

We also need a constitutive equation which links the temperature field, $u$, to the heat
flux, $\mathbf{q}$. The simplest case is to use Fourier's law

```math
\mathbf{q} = -k \mathbf{\nabla}u
```

where $k$ is the conductivity of the material. For simplicity we will consider only
constant conductivity $k$.

## Weak format

The solution to the equation above is usually calculated from the corresponding weak
format. By multiplying the equation with an arbitrary test function $\delta u$, integrating
over the domain and using partial integration we obtain the *weak form*;
Find $u \in \mathbb{U}$ s.t.

```math
\int_\Omega \mathbf{\delta u} \cdot (k \mathbf{u}) \mathrm{d}\Omega =
\int_{\Gamma_\mathrm{N}} \delta u q^\mathrm{p} \mathrm{d}\Gamma +
\int_\Omega \delta u b \mathrm{d}\Omega \quad \forall \delta u \in \mathbb{U}^0
```

where $\mathbb{U}, \mathbb{U}^0$ are function spaces with sufficiently regular functions.
It can be shown that the solution to the weak form is identical to the solution to the
strong format.

## FE-approximation

We now introduce the finite element approximation $u_h \approx u$ as a sum of shape
functions, $N_i$ and nodal values, $a_i$. We approximate the test function in the same way
(known as the Galerkin method):

```math
u_\mathrm{h} = \sum_{i=1}^{\mathrm{N}} N_i a_i,\qquad
\delta u_\mathrm{h} = \sum_{i=1}^{\mathrm{N}} N_i \delta a_i
```

We may now inserted these approximations in the weak format, which results in

```math
\sum_i^N \sum_j^N \delta a_i \int_\Omega \mathbf{\nabla} N_i \cdot (k \cdot \mathbf{\nabla} N_j) \mathrm{d}\Omega a_j =
\sum_i^N \delta a_i \int_\Gamma N_i q^\mathrm{p} \mathrm{d}\Gamma +
\sum_i^N \delta a_i \int_\Omega N_i b \mathrm{d}\Omega
```

Since $\delta u$ can be chosen arbitrary, the nodal values $\delta a_i$ can be chosen
arbitrary. Thus, the equation can be written as a linear system of equations

```math
\underline{K}\ \underline{a} = \underline{f}
```

where $\underline{K}$ is the (tangent) stiffness matrix, $\underline{a}$ is the solution
vector with the nodal values and $\underline{f}$ is the force vector. The elements of
$\underline{K}$ and $\underline{f}$ are given by

```math
\underline{K}_{ij} =
    \int_\Omega \mathbf{\nabla}N_i \cdot (k \cdot \mathbf{\nabla}N_j) \mathrm{d}\Omega\\

\underline{f}_{i} =
    \int_\Gamma N_i q^\mathrm{p} \mathrm{d}\Gamma + \int_\Omega N_i b \mathrm{d}\Omega
```

The solution to the system (which in this case is linear) is simply given by inverting the
matrix $\underline{K}$. We also need to take care of the Dirichlet boundary conditions, by
enforcing the correct nodal values $a_i$ to the prescribed values.

```math
\underline{a} = \underline{K}^{-1}\ \underline{f}
```

## Implementation

In practice, the shape functions $N$ are only non-zero on parts of the domain $\Omega$.
Thus, the integrals are evaluated on sub-domains, called *elements* or *cells*. Each cell
gives a contribution to the global stiffness matrix and force vector. For a solution of the
heat equation, as implemented in `JuAFEM`, check out
[this thoroughly commented example](@ref Heat-Equation).
