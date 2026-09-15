# How-to guides

This page gives an overview of the *how-to guides*. How-to guides address various common
tasks one might want to do in a finite element program. Many of the guides are extensions,
or build on top of, the tutorials and, therefore, some familiarity with Ferrite is assumed.

---

#### [Postprocessing and visualization](postprocessing.md)

This guide builds on top of [Tutorial 1: Heat equation](../tutorials/heat_equation.md) and
discusses various post processing techniques with the goal of visualizing primary fields
(the finite element solution) and secondary quantities (e.g. fluxes, stresses, etc.).
Concretely, this guide answers:
 - How to visualize data from quadrature points?
 - How to evaluate the finite element solution, or secondary quantities, in arbitrary points
   of the domain?

---

#### [Multithreaded assembly](threaded_assembly.md)

This guide modifies [Tutorial 2: Linear elasticity](../tutorials/linear_elasticity.md) such
that the program is using multi-threading to parallelize the assembly procedure. Concretely
this shows how to use grid coloring and "scratch values" in order to use multi-threading
without running into race-conditions.

---

#### [GPU assembly](gpu_assembly.md)

This guide builds on top of [Tutorial 1: Heat equation](../tutorials/heat_equation.md) such
that the program is using CUDA to parallelize the assembly procedure. Concretely
this shows how to use grid coloring and the structure-of-arrays types in Ferrite.

---

#### [Matrix-free operator evaluation](matrix_free.md)

This guide implements the operator of [Tutorial 1: Heat equation](../tutorials/heat_equation.md)
*matrix-free*: instead of assembling a sparse matrix, only a small tensor per quadrature
point is stored ("partial assembly") and the matrix-vector product is evaluated cell by
cell with sum factorization, exploiting the tensor product structure of the hexahedral
basis and quadrature. This uses much less memory than the assembled sparse matrix, in
particular for higher order interpolations.

---
