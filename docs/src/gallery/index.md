# Code gallery

This page gives an overview of the code gallery. Compared to the tutorials, these programs
do not focus on teaching Ferrite, but rather focus on showing how Ferrite can be used "in
the wild".

!!! note "Contribute to the gallery!"
    Most of the gallery is user contributed. If you use Ferrite, and have something you want
    to share, please contribute to the gallery! This could, for example, be your research
    code for a published paper, some interesting application, or just some nice trick.

---

#### [Helmholtz equation](helmholtz.md)

Solves the Helmholtz equation on the unit square using a combination of Dirichlet and
Neumann boundary conditions and the method of manufactured solutions.

*Contributed by*: Kristoffer Carlsson ([@KristofferC](https://github.com/KristofferC)).

---

#### [Nearly incompressible hyperelasticity](quasi_incompressible_hyperelasticity.md)

This program combines the ideas from [Tutorial 3: Incompressible
elasticity](../tutorials/incompressible_elasticity.md) and [Tutorial 4:
Hyperelasticity](../tutorials/incompressible_elasticity.md) to construct a mixed element
solving three-dimensional displacement-pressure equations.

*Contributed by*: Bhavesh Shrimali ([@bhaveshshrimali](https://github.com/bhaveshshrimali)).

---

#### [Ginzburg-Landau model energy minimization](landau.md)

A basic Ginzburg-Landau model is solved.
[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) is used to compute the
gradient and hessian of the energy function. Multi-threading is used to parallelize the
assembly procedure.

*Contributed by*: Louis Ponet ([@louisponet](https://github.com/louisponet)).

---

#### [Topology optimization](topology_optimization.md)

Topology optimization is shown for the bending problem by using a SIMP material model. To
avoid numerical instabilities, a regularization scheme requiring the calculation of the
Laplacian is imposed, which is done by using the grid topology functionalities.

*Contributed by*: Mischa Blaszczyk ([@blaszm](https://github.com/blaszm)).
