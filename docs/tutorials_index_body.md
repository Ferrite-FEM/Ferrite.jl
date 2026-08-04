# Tutorials

On this page you find an overview of Ferrite tutorials. The tutorials explain and show how
Ferrite can be used to solve a wide range of problems. See also the [Code
gallery](../gallery/index.md) for more examples.

The tutorials all follow roughly the same structure:
 - **Introduction** introduces the problem to be solved and discusses the learning outcomes
   of the tutorial.
 - **Commented program** is the code for solving the problem with explanations and comments.
 - **Plain program** is the raw source code of the program.

When studying the tutorials it is a good idea to obtain a local copy of the code and run it
on your own machine as you read along. Some of the tutorials also include suggestions for
tweaks to the program that you can try out on your own.

### Tutorial index

The tutorials are listed in roughly increasing order of complexity. However, since they
focus on different aspects, and solve different problems, it is suggested to have a look at
the brief descriptions below to get an idea about what you will learn from each tutorial.

If you are new to Ferrite then Tutorials 1 - 7 are the best place to start. These
tutorials introduce and teach most of the basic finite element techniques (e.g. linear
and non-linear problems, scalar- and vector-valued problems, Dirichlet and Neumann boundary
conditions, mixed finite elements, time integration, direct and iterative linear solvers,
etc). In particular the very first tutorial is essential in order to be able to follow any
of the other tutorials. The remaining tutorials discuss more advanced topics.

---

#### [Tutorial 1: Heat equation](heat_equation.md)

This tutorial guides you through the process of solving the linear stationary heat equation
(i.e. Poisson's equation) on a unit square with homogeneous Dirichlet boundary conditions.
This tutorial introduces and teaches many important parts of Ferrite: problem setup, degree
of freedom management, assembly procedure, boundary conditions, solving the linear system,
visualization of the result. *Understanding this tutorial is essential to follow more
complex tutorials.*

**Keywords**: scalar-valued solution, Dirichlet boundary conditions.

---

#### [Tutorial 2: Linear elasticity](linear_elasticity.md)

This tutorial guides you through the process of solving the linear elasticity problem. The tutorial builds on the concepts from Tutorial 1 and introduces vector-valued solutions and Neumann boundary conditions.

**Keywords**: vector-valued solution, Dirichlet and Neumann boundary conditions.

---

#### [Tutorial 3: Elastodynamics and modal analysis](@ref tutorial-elastodynamics)

This tutorial extends [Tutorial 2: Linear elasticity](linear_elasticity.md) from statics to
dynamics: the natural frequencies and vibration modes of a cantilever beam are computed by
solving a generalized eigenvalue problem, and the free vibration after releasing a static
tip load is simulated using Newmark time integration. Both parts are verified against
analytical beam theory.

**Keywords**: structural dynamics, mass matrix, generalized eigenvalue problem, natural
frequencies, modal analysis, Rayleigh damping, Newmark time integration.

---

#### [Tutorial 4: Incompressible elasticity](incompressible_elasticity.md)

This tutorial focuses on a mixed formulation of linear elasticity, with (vector)
displacement and (scalar) pressure as the two unknowns, suitable for incompressibility.
Thus, this tutorial guides you through the process of solving a problem with two unknowns
from two coupled weak forms. The problem that is studied is Cook's membrane in the
incompressible limit.

**Keywords**: mixed finite elements, Dirichlet and Neumann boundary conditions.

---

#### [Tutorial 5: Hyperelasticity](hyperelasticity.md)

In this tutorial you will learn how to solve a non-linear finite element problem. In
particular, a hyperelastic material model, in a finite strain setting, is used to solve the
rotation of a cube. Automatic differentiation (AD) is used for the constitutive relations.
Newton's method is used for the non-linear iteration, and a conjugate gradient (CG) solver
is used for the linear solution of the increment.

**Keywords**: non-linear finite element, finite strain, automatic differentiation (AD),
Newton's method, conjugate gradient (CG).

---

#### [Tutorial 6: von Mises Plasticity](plasticity.md)

This tutorial revisits the cantilever beam problem from [Tutorial 2: Linear
elasticity](linear_elasticity.md), but instead of linear elasticity a plasticity model is
used for the constitutive relation. You will learn how to solve a problem which require the
solution of a local material problem, and the storage of material state, in each quadrature
point. Newton's method is used both locally in the material routine, and globally on the
finite element level.

**Keywords**: non-linear finite element, plasticity, material modeling, state variables,
Newton’s method.

---

#### [Tutorial 7: Transient heat equation](@ref tutorial-transient-heat-equation)

In this tutorial the transient heat equation is solved on the unit square. The problem to be
solved is thus similar to the one solved in the first tutorial, [Heat
equation](heat_equation.md), but with time-varying boundary conditions. In particular you
will learn how to solve a time dependent problem with an implicit Euler scheme for the time
integration.

**Keywords**: time dependent finite elements, implicit Euler time integration.

---

#### [Tutorial 8: Computational homogenization](computational_homogenization.md)

This tutorial guides you through computational homogenization of a representative volume
element (RVE) consisting of a soft matrix material with stiff inclusions. The computational
mesh is read from an external mesh file generated with Gmsh. Dirichlet and periodic boundary
conditions are used.

**Keywords**: Gmsh mesh reading, Dirichlet and periodic boundary conditions.

---

#### [Tutorial 9: Stokes flow](stokes-flow.md)

In this tutorial Stokes flow with (vector) velocity and (scalar) pressure is solved on a
quarter circle. Rotationally periodic boundary conditions is used for the inlet/outlet
coupling. To obtain a unique solution, a mean value constraint is applied on the pressure
using an affine constraint. The computational mesh is generated directly using the Gmsh API.

**Keywords**: periodic boundary conditions, mean value constraint, mesh generation with
Gmsh.

---

#### [Tutorial 10: Porous media (SubDofHandler)](porous_media.md)

This tutorial introduces how to solve a complex linear problem, where there are different
fields on different subdomains, and different cell types in the grid. This requires using
the `SubDofHandler` interface.

**Keywords**: Mixed grids, multiple fields, porous media, `SubDofHandler`.

---

#### [Tutorial 11: Incompressible Navier-Stokes equations](ns_vs_diffeq.md)

In this tutorial the incompressible Navier-Stokes equations are solved. The domain is
discretized in space with Ferrite as usual, and then formulated in a way to be compatible
with the [OrdinaryDiffEq.jl](https://docs.sciml.ai/OrdinaryDiffEq/) package, which is used
for the time-integration.

**Keywords**: non-linear time dependent problem.

---

#### [Tutorial 12: Reactive surface](@ref tutorial-reactive-surface)

In this tutorial a reaction diffusion system on a sphere surface embedded in 3D is solved.
Ferrite is used to assemble the diffusion operators and the mass matrices. The problem is
solved by using the usual first order reaction diffusion operator splitting.

**Keywords**: embedded elements, operator splitting, gmsh.

---

#### [Tutorial 13: Linear shell](@ref tutorial-linear-shell)

In this tutorial a linear shell element formulation is set up as a two-dimensional domain
embedded in three-dimensional space. This will teach, and perhaps inspire, you on how
Ferrite can be used for non-standard things and how to add "hacks" that build on top of
Ferrite.

**Keywords**: shell elements, automatic differentiation.

---

#### [Tutorial 14: Discontinuous Galerkin heat equation](@ref tutorial-dg-heat-equation)

This tutorial guides you through the process of solving the linear stationary heat equation
(i.e. Poisson's equation) on a unit square with inhomogeneous Dirichlet and Neumann boundary
conditions using the interior penalty discontinuous Galerkin method. This tutorial follows
the [heat equation tutorial](@ref tutorial-heat-equation), introducing facet and interface
iterators, jump and average operators, and cross-element coupling in sparsity patterns. This
example was developed as part of the *Google Summer of Code* funded project ["Discontinuous
Galerkin Infrastructure For the finite element toolbox
Ferrite.jl"](https://summerofcode.withgoogle.com/programs/2023/projects/SLGbRNI5).

**Keywords**: scalar-valued solution, Dirichlet boundary conditions, Discontinuous Galerkin, Interior penalty.

---

#### [Tutorial 15: Heat equation with adaptive mesh refinement](@ref tutorial-heat-adaptivity)

In this tutorial the heat equation is solved on a cube with adaptive mesh refinement (AMR).
A manufactured solution with a sharp spherical feature drives the refinement, using a
Zienkiewicz-Zhu error estimator and Dörfler marking. The hanging nodes introduced by the
refinement are constrained to keep the approximation conforming.

**Keywords**: adaptive mesh refinement, error estimation, non-conforming grid, hanging nodes.

---

#### [Tutorial 16: Darcy flow with H(div) elements](@ref tutorial-darcy-flow)

In this tutorial Darcy flow through a domain with an almost impermeable barrier is solved
using the mixed form of the Poisson problem, with the flux as an explicit unknown
discretized by H(div)-conforming Raviart-Thomas elements paired with a discontinuous
pressure. Pressure boundary conditions enter weakly, while the normal flux is prescribed
with `ProjectedDirichlet`. The local (element-wise) mass conservation of the mixed method
is verified and compared against the standard primal formulation.

**Keywords**: mixed finite elements, H(div), Raviart-Thomas, weak (natural) boundary
conditions, `ProjectedDirichlet`, local conservation.

---

#### [Tutorial 17: Bidomain model](@ref tutorial-bidomain)

In this tutorial the bidomain model of cardiac electrophysiology, coupled to the
FitzHugh-Nagumo cell model, is solved on a square patch of tissue. The three coupled fields
are discretized in space with Ferrite, which yields a differential-algebraic system in mass
matrix form, and integrated in time with an adaptive stiff solver from
[DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl). The solution
develops into a spiral wave.

**Keywords**: multiple coupled fields, block matrices, mass matrix form, DAE,
DifferentialEquations.jl, anisotropic diffusion.
