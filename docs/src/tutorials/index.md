```@meta
DocTestSetup = :(using Ferrite)
```

# Tutorials

On this page, you find an overview of Ferrite tutorials. The collection contains programs displaying different aspects and techniques of the `Ferrite` toolbox.
The programs are sorted as

* A complete list shortly summarizing the topic, what they teach and which keywords related to the task they contain
* Grouped by topic

## List of examples (sequential order)
1. [Heat Equation](@ref tutorial-heat-equation)

   This example presents the easiest way to get started. The heat equation is solved on a unit square. For this purpose, a program is set up containing the core aspects of finite element computation: weak and strong forms, trial and test functions, degrees of freedom, boundary conditions, element routine, global assembly, solving the system and visualizing the results.

   Keywords: Fundamentals, heat equation, weak and strong form, Dirichlet boundary condition, assembly

2. [Postprocessing](@ref tutorial-postprocessing)

   Based on example 1, visualization of the flux variable is done via the L2Projector. Additionally, the point evaluation along a line is shown.

   Keywords: Fundamentals, postprocessing, heat equation, flux, L2-projection, point evaluation

3. [Helmholtz Equation](@ref tutorial-helmholtz)

   The Helmholtz equation is solved on a unit square. Dirichlet and Neumann boundary conditions are applied at different parts of the boundary. A known analytical solution is approximately reproduced to verify the finite element solution.

   Keywords: Fundamentals, Helmholtz equation, Dirichlet boundary condition, Neumann boundary condition, verification, boundary integrals

4. [Incompressible Elasticity](@ref tutorial-incompressible-elasticity)

   A mixed element for solving unidirectional displacement-pressure coupling is constructed. The solution for nearly incompressible materials is compared for different interpolations.

   Keywords: Mixed elements, multiple fields, unidirectional coupling, displacement-pressure equation, incompressibility, stability

5. [Hyperelasticity](@ref tutorial-hyperelasticity)

   A hyperelastic material model is solved in a finite strain setting. Stress and material tangent are recovered by using automatic differentiation. Newton’s method is used to iteratively solve the resulting non-linear system.

   Keywords: Non-linear problem, hyperelasticity, finite strain, large deformations, Newton's method, conjugate gradient, automatic differentiation

6. [Threaded Assembly](@ref tutorial-threaded-assembly)

   This example shows the threaded assembly of a finite element stiffness matrix, i.e. the calculation is speed up by using parallelization. Different colorings of two-dimensional meshes are shown to visualize the split of the mesh in such a way that no threads interfere with each other.

   Keywords: Parallelization, performance, threads, coloring

7. [von Mises Plasticity](@ref tutorial-plasticity)

   A cantilever beam is solved applying a plasticity material model, requiring a constitutive driver. Handling of state, flux and internal variables within elements is shown. Newton’s method is used to iteratively solve the non-linear system.

   Keywords: Plasticity, 3D, material modeling, material state, non-linear problem, Newton’s method,

8. [Time Dependent Problems](@ref tutorial-transient-heat-equation)

   The transient heat equation is solved on a rectangular plate. The time discretization is done by using the implicit Euler scheme.

   Keywords: Time dependent problem, transient heat equation, implicit Euler scheme

9. [Ginzburg-Landau model energy minimization](@ref tutorial-ginzburg-landau-minimizer)

   A basic Ginzburg-Landau model is solved by combining `Ferrite` with `ForwardDiff`. Using threads, the calculation time is optimized.

   Keywords: Ginzburg-Landau, ForwardDiff.jl, parallelization, optimization, performance

10. [Linear shell](@ref tutorial-linear-shell)

   A program for the use of linear shell elements is set up. The theoretical background is introduced as well.

   Keywords: Shell elements, displacements, rotations, multiple fields, ForwardDiff.jl, reduced integration

11. [Nearly Incompressible Hyperelasticity](@ref tutorial-nearly-incompressible-hyperelasticity)

   This program combines ideas from the examples Incompressible Elasticity and Hyperelasticity to construct a mixed element solving three-dimensional displacement-pressure equations.

   Keywords: Non-linear problem, hyperelasticity, finite strain, large deformations, Newton's method, automatic differentiation, coupled problem, mixed elements, multiple fields, displacement-pressure equation, incompressibility

12. [Computational homogenization](@ref tutorial-computational-homogenization)

   A two-dimensional representative volume element (RVE) is loaded in shear. Dirichlet and periodic boundary conditions are applied and the results from the homogenization are compared to the Voigt and Reuss bounds.

   Keywords: homogenization, periodic boundary conditions, representative volume element (RVE), microscale, Voigt and Reuss bound

13. [Stokes flow](@ref tutorial-stokes-flow)

   Stokes flow on a quarter circle is solved by applying periodic boundary conditions. The weak and strong form of the problem are discussed including constrains. Mesh generation is done directly via the Gmsh API.

   Keywords: Periodic boundary conditions, multiple fields, mean value constraint, mesh generation with Gmsh, coupled problem, weak and strong form

14. [Topology Optimization](@ref tutorial-topology-optimization)

   Topology optimization is shown for the bending problem by using a SIMP material model. To avoid numerical instabilities, a regularization scheme requiring the calculation of the Laplacian is imposed, which is done by using the grid topology functionalities.

   Keywords: Topology optimization, weak and strong form, non-linear problem, Laplacian, grid topology

## Other examples
1. [Incompressible Navier-Stokes Equations via DifferentialEquations.jl](@ref tutorial-ins-ordinarydiffeq)

   The weak form of the semi-discretized incompressible Navier-Stokes equations are derived from the strong form and implemented in the finite element code. Then, the time-dependent solution is calculated by using the DifferentialEquations.jl package. For this purpose, the PDE is required in a specific form.

   Keywords: Fluid dynamics, weak and strong form, solver, incompressibility, time-dependent problem

## Grouped by topic
* Fundamentals: [Heat Equation](@ref tutorial-heat-equation), [Postprocessing](@ref tutorial-postprocessing), [Helmholtz Equation](@ref tutorial-helmholtz), [Incompressible Elasticity](@ref tutorial-incompressible-elasticity)
* Non-linear Problems: [Hyperelasticity](@ref tutorial-hyperelasticity), [von Mises Plasticity](@ref tutorial-plasticity), [Nearly Incompressible Hyperelasticity](@ref tutorial-nealy-incompressibly-hyperelasticity), [Computational homogenization](@ref tutorial-computational-homogenization), [Topology Optimization](@ref tutorial-topology-optimization)
* Time dependent problems: [Time Dependent Problems](@ref tutorial-transient-heat-equation), [Incompressible Navier-Stokes Equations via DifferentialEquations.jl](@ref tutorial-ins-ordinarydiffeq)
* Multiple fields: [Incompressible Elasticity](@ref tutorial-incompressible-elasticity), [Linear shell](@ref tutorial-linear-shell), [Nearly Incompressible Hyperelasticity](tutorial-nearly-incompressible-hyperelasticity), [Stokes flow](@ref tutorial-stokes-flow)
* Periodic problems: [Computational homogenization](@ref tutorial-computational-homogenization), [Stokes flow](@ref tutorial-stokes-flow)
* Advanced: [Threaded Assembly](@ref tutorial-threaded-assembly), [Ginzburg-Landau model energy minimization](@ref tutorial-ginzburg-landau-minimizer)
