# How-to guides

This page gives an overview of the *how-to guides*. How-to guides address various common
tasks one might want to do in a finite element program. Many of the guides are extensions,
or build on top of, the tutorials and, therefore, some familiarity with Ferrite is assumed.

---

#### [Post processing and visualization](../postprocessing/)

This guide builds on top of [Tutorial 1: Heat equation](../../tutorials/heat_equation/) and
discusses various post processsing techniques with the goal of visualizing primary fields
(the finite element solution) and secondary quantities (e.g. fluxes, stresses, etc.).
Concretely, this guide answers:
 - How to visualize data from quadrature points?
 - How to evaluate the finite element solution, or secondary quantities, in arbitrary points
   of the domain?

---

#### [Multi-threaded assembly](../threaded_assembly/)

This guide modifies [Tutorial 2: Linear elasticity](../../tutorials/linear_elasticity/) such
that the program is using multi-threading to parallelize the assembly procedure. Concretely
this shows how to use grid coloring and "scratch values" in order to use multi-threading
without running into race-conditions.

---
