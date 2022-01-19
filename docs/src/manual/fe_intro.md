# Introduction to FEM

Here we will present a very brief introduction to partial differential equations (PDEs) and
to the finite element method (FEM). Perhaps the simplest PDE of all is the (steady-state, linear)
heat equation, also known as the Poisson equation. We will use this equation as a demonstrative
example of the method, and demonstrate how we go from the strong form of the equation, to
the weak form, and then finally to the discrete FE problem.

## Strong Form

TODO: Image of domain with boundary annotation and boundary normal vector.

The strong form of the heat equation may be written as:

```math
- \nabla \cdot \mathbf{q}(u) = b \quad \forall \, x \in \Omega,
```

where $u$ is the unknown temperature field, $\mathbf{q}$ is the heat flux, $b$ is an
internal heat source, and $\Omega$ is the domain on which the equation is defined. To
complete the problem we need to specify what happens at the domain boundary $\Gamma$.
This set of specifications is called *boundary conditions*. There are different types of
boundary conditions, where the most common ones are Dirichlet -- which means that the solution
$u$ is known at some part of the boundary, and Neumann -- which means that the gradient
of the solution, $\nabla$ is known. Formally we write for our example

```math
u = u^\mathrm{p} \quad \forall \, \mathbf{x} \in \Gamma_\mathrm{D},\\
\mathbf{q} \cdot \mathbf{n} = q^\mathrm{p} \quad \forall \, \mathbf{x} \in \Gamma_\mathrm{N},
```

i.e. the temperature is prescribed to a known function $u^\mathrm{p}$ at the Dirichlet part
of the boundary, $\Gamma_\mathrm{D}$, and the heat flux is prescribed to $q^\mathrm{p}$ at
the Neumann part of the boundary, $\Gamma_\mathrm{N}$, where $\mathbf{n}$ describes the outward
pointing normal vector at the boundary.

We also need a constitutive equation which links the temperature field, $u$, to the heat
flux, $\mathbf{q}$. The simplest case is to use Fourier's law

```math
\mathbf{q}(u) = -k \nabla u
```

where $k$ is the conductivity of the material. For simplicity we will consider only
constant conductivity $k$.

## Weak Form

The solution to the equation above is usually calculated from the corresponding weak
form. By multiplying the equation with an arbitrary test function $\delta u$, integrating
over the domain and using partial integration we obtain the *weak form*. Now our problem
can be stated as:

Find $u \in \mathbb{U}$ s.t.

```math
\int_\Omega \nabla \delta u \cdot (k \nabla u) \, \mathrm{d}\Omega =
\int_{\Gamma_\mathrm{N}} \delta u \, q^\mathrm{p} \, \mathrm{d}\Gamma +
\int_\Omega \delta u \, b \, \mathrm{d}\Omega \quad \forall \, \delta u \in \mathbb{T}
```

where $\mathbb{U}, \mathbb{T}$ are suitable function spaces with sufficiently regular
functions. Under very general assumptions it can be shown that the solution to the weak
form is identical to the solution to the strong form.

## Finite Element Approximation

TODO: Image of geometric discretization with triangles.

Using the finite element method to solve partial differential equations is usually
preceded with the construction of a discretization of the domain $\Omega$ into a finite
set of *elements* or *cells*. We call this geometric discretization *grid* (or *mesh*)
and denote it with $\Omega_h$. In this example the corners of the triangles are called
*nodes*.

Next we introduce the finite element approximation $u_\mathrm{h} \approx u$ as a sum of N nodal
shape functions, where we denote each of these function by $N_i$ and the corresponding nodal
values $a_i$. In this example we choose to approximate the test function in the same way. This
approach is known as the *Galerkin finite element method*. Formally we write the evaluation
of our approximations at a specific point $\mathbf{x}$ in our domain $\Omega$ as:

```math
u_\mathrm{h}(\mathbf{x}) = \sum_{i=1}^{\mathrm{N}} N_i(\mathbf{x}) \, a_i,\qquad
\delta u_\mathrm{h}(\mathbf{x}) = \sum_{i=1}^{\mathrm{N}} N_i(\mathbf{x}) \, \delta a_i \, .
```

In the following the argument $\mathbf{x}$ is dropped to keep the notation compact.
We may now inserted these approximations in the weak form, which results in

```math
\sum_j^N \left(\sum_i^N \delta a_i \int_{\Omega_\mathrm{h}} \nabla N_i \cdot (k \nabla N_j) \, \mathrm{d}\Omega \right) a_j =
\sum_i^N \delta a_i \int_{\Gamma_\mathrm{N}} N_i \, q^\mathrm{p} \, \mathrm{d}\Gamma +
\sum_i^N \delta a_i \int_{\Omega_\mathrm{h}} N_i \, b \, \mathrm{d}\Omega \, .
```

Since this equation must hold for arbitrary $\delta u_\mathrm{h}$, the equation must especially
hold for the specific choice that only one of the nodal values $\delta a_i$ is fixed to 1 while
an all other coefficients are fixed to 0. Repeating this argument for all $i$ from 1 to N we obtain
N linear equations. This way the discrete problem can be written as a system of linear equations

```math
\underline{K}\ \underline{a} = \underline{f} \, ,
```

where we call $\underline{K}$ the (tangent) *stiffness matrix*, $\underline{a}$ the *solution
vector* with the nodal values and $\underline{f}$ the *force vector*. The specific naming is for
historical reasons, because the finite element method has its origins in mechanics. The elements
of $\underline{K}$ and $\underline{f}$ are given by

```math
\underline{K}_{ij} =
    \int_{\Omega_\mathrm{h}} \nabla N_i \cdot (k \nabla N_j) \mathrm{d}\Omega \, , \\

\underline{f}_{i} =
    \int_{\Gamma_\mathrm{N}} N_i \, q^\mathrm{p} \, \mathrm{d}\Gamma + \int_{\Omega_\mathrm{h}} N_i \, b \, \mathrm{d}\Omega \, .
```

Finally we also need to take care of the Dirichlet boundary conditions. These are enforce by
setting the corresponding $a_i$ to the prescribed values and eliminating the associated equations
from the system. Now, solving this equation system yields an the nodal values and thus an
approximation to the true solution.

## Notes on the Implementation

In practice, the shape functions $N_i$ are only non-zero on parts of the domain $\Omega_\mathrm{h}$.
Thus, the integrals are evaluated on sub-domains, called *elements* or *cells*.

TODO: Image of a linear basis function on the grid above.

Each cell gives a contribution to the global stiffness matrix and force vector. The process
of constructing the system of equations is also called *assembly*. For clarification,
let us rewrite the formula for the stiffness matrix entries as follows:
```math
\underline{K}_{ij}
    = \int_{\Omega_\mathrm{h}} \nabla N_i \cdot (k \nabla N_j) \mathrm{d}\Omega
    = \sum_{E \in \Omega_\mathrm{h}} \int_E \nabla N_i \cdot (k \nabla N_j) \mathrm{d}\Omega \, .
```
This formulation underlines the element-centric perspective of finite element methods and
reflects how it is usually implemented in software. For an example of the implementation to
solve a heat problem with `Ferrite` check out [this thoroughly commented example](@ref Heat-Equation).

## More Details

We finally want to note that this quick introduction barely scratches the surface of the finite element
method. Also, we presented some things in a simplified way for the sake of keeping this article short
and concise. There is a large corpus of literature and online tutorials containing more details about
the finite element method. To give a few recommendations there is:
* [Hans Petter Langtangen's Script](http://hplgit.github.io/INF5620/doc/pub/sphinx-fem/index.html)
* [Wolfgang Bangerth's Lecture Series](https://www.math.colostate.edu/~bangerth/videos.html)
* *The Finite Element Method for Elliptic Problems* by Philippe Ciarlet
* *Finite Elements: Theory, Fast Solvers, and Applications in Elasticity Theory* by Dietrich Braess
* *An Analysis of the Finite Element Method* by Gilbert Strang and George Fix
* *Finite Element Procedures* by K.J. Bathe
* *The Finite Element Method: Its Basis and Fundamentals* by Olek Zienkiewicz, Robert Taylor and J.Z. Zhu
* *Higher-Order Finite Element Methods* by Pavel Šolín, Karel Segeth and Ivo Doležel
This list is neither meant to be exhaustive, nor does the absence of a work mean that it is in any way
bad or not recommendable.