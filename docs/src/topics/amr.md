# Adaptive Mesh Refinement

Adaptive mesh refinement (AMR) is a computational technique used in finite element analysis to enhance the accuracy and efficiency of simulations.
It involves dynamically adjusting the mesh resolution based on some criteria.
By refining the mesh in regions where the solution exhibits features of interest, AMR ensures that computational resources are concentrated where they are most needed, leading to more accurate results without a proportional increase in computational cost.
This refinement can be achieved in different fashions by either adjusting the mesh size (h-adaptivity), the polynomial order of the Ansatz functions (p-adaptivity) or the nodal positions (r-adaptivity).

In Ferrite.jl, adaptivity is achieved through a p4est type of implementation which covers h-adaptivity.
This approach is designed to handle unstructured hexahedral (in 3D) and quadrilateral (in 2D) meshes.
A further restriction of the p4est type of implementation is isotropic refinement, meaning that elements are subdivided into smaller elements of the same shape.

In AMR different phenomena and vocabulary emerge which we group into the following aspects

- hanging nodes
- balancing
- error estimation

## Hanging Nodes
### What Are Hanging Nodes?

Hanging nodes occur during the process of mesh refinement.
When a mesh is refined, some elements may be subdivided while their neighboring elements are not.
This results in nodes that lie on the edges or faces of coarser elements but do not coincide with the existing nodes of those elements.
These intermediate nodes are referred to as hanging nodes.

Consider the following situation:
```
x-----------x-----------x               x-----------x-----------x
|7          |8          |9              |7          |8          |9
|           |           |               |           |           |
|           |           |               |           |           |
|           |           |               |           |           |
|           |           |               |           |           |
|           |           |               |           |           |
|           |           |   refinement  |           |           |
x-----------x-----------x   -------->   x-----x-----x-----------x
|4          |5          |6              |4    |14   |5          |6
|           |           |               |     |     |           |
|           |           |               |     |     |           |
|           |           |               x-----x-----x           |
|           |           |               |11   |12   |13         |
|           |           |               |     |     |           |
|           |           |               |     |     |           |
x-----------x-----------x               x-----x-----x-----------x
 1           2           3               1     10     2           3
```
The new introduced nodes 10, 11 and 12 are shared and therefore are not hanging nodes.
However, the nodes 14 and 13 are not shared with the neighboring coarser element and therefore hanging.

### Implications of Hanging Nodes

The presence of hanging nodes poses the challenge of non-conformity:
A mesh with hanging nodes is non-conforming because the finite element mesh no longer adheres to the requirement that elements meet only at their nodes or along entire edges or faces.
This lack of conformity can lead to difficulties in maintaining the continuity of the solution across the mesh.
However we can recover the continuity of the solution by constraining the hanging nodes.


### How to Treat Hanging Nodes

To address the issues introduced by hanging nodes, specific strategies and constraints are employed.
The degrees of freedom (DoFs) associated with hanging nodes are constrained based on the surrounding coarser mesh elements.
For example, in a linear finite element method, the value at a hanging node can be constrained to be the average of the values at the adjacent vertices of the coarser element.
As for the example above node 13 could be constrained to $\boldsymbol{u}[13]=0.5\boldsymbol{u}[5]+0.5\boldsymbol{u}[2]$.
As soon as higher polynomial degrees are involved, things become more involved.
In Ferrite, a conformity constraint can be constructed with the ConstraintHandler when using a DofHandler which has been constructed with a grid passed from `Ferrite.creategrid(adaptive_grid::ForestBWG)`.
This conformity constraint ensures that each hanging node is constrained appropriately.

```julia
ch = ConstraintHandler(dh)
add!(ch, Ferrite.ConformityConstraint(:u))
```

## Balancing

Hanging nodes can depend in a nested fashion on other hanging nodes.
This case complicates the `ConformityConstraint` massively and is therefore often prohibited by the act of balancing.
Mesh balancing refers to the process of ensuring a smooth and gradual transition in element sizes across the computational mesh.
This is especially important in adaptive mesh refinement (AMR), where different regions of the mesh may undergo varying levels of refinement based on the solution's characteristics.
The goal of mesh balancing is to prevent isolated regions of excessive refinement, which can lead to inefficiencies in terms of constructing conformity constraints and numerical instability.
A famous balancing approach is the 2:1 balance, where it is ensured that a hanging node only depends on non-hanging nodes.
Therefore, exactly one level of non-conformity is allowed.
An example of this process is visualized below.

```
x-----------x-----------x               x-----------x-----------x
|           |           |               |     |     |           |
|           |           |               |     |     |           |
|           |           |               |     |     |           |
|           |           |               |-----x-----|           |
|           |           |               |     |     |           |
|           |           |               |     |     |           |
|           |           |   balancing   |     |     |           |
x-----x--x--x-----------x   -------->   x-----x--x--x-----------x
|     |  |  |           |               |     |  |  |     |     |
|     x--x--x           |               |     x--x--x     |     |
|     |  |  |           |               |     |  |  |     |     |
x-----x--x--x           |               x-----x--x--x-----x-----|
|     |     |           |               |     |     |     |     |
|     |     |           |               |     |     |     |     |
|     |     |           |               |     |     |     |     |
x-----x-----x-----------x               x-----x-----x-----------x
```

Note that in the example above, the top right element hasn't been refined.
However, in some cases it is advantageous to do so in order to have a smoother transition in element size.
Therefore, by default, the adaptive mesh is also refined over this vertex, leading to the following result:

```
x-----------x-----------x               x-----------x-----------x
|           |           |               |     |     |     |     |
|           |           |               |     |     |     |     |
|           |           |               |     |     |     |     |
|           |           |               |-----x-----|-----x-----|
|           |           |               |     |     |     |     |
|           |           |               |     |     |     |     |
|           |           |   balancing   |     |     |     |     |
x-----x--x--x-----------x   -------->   x-----x--x--x-----------x
|     |  |  |           |               |     |  |  |     |     |
|     x--x--x           |               |     x--x--x     |     |
|     |  |  |           |               |     |  |  |     |     |
x-----x--x--x           |               x-----x--x--x-----x-----|
|     |     |           |               |     |     |     |     |
|     |     |           |               |     |     |     |     |
|     |     |           |               |     |     |     |     |
x-----x-----x-----------x               x-----x-----x-----------x
```

In Ferrite's p4est implementation, one can call `balanceforest!` to balance the adaptive grid.

```julia
Ferrite.balanceforest!(adaptive_grid)
```

## Error Estimation
Error estimation is a critical component of adaptive mesh refinement (AMR) in finite element analysis.
The primary objective of error estimation is to identify regions of the computational domain where the numerical solution is less accurate and requires further refinement.
Accurate error estimation guides the adaptive refinement process, ensuring that computational resources are concentrated in areas where they will have the most significant impact on improving solution accuracy.

In practice, error estimators evaluate the local error in the finite element solution by comparing it to an approximation of the true solution.
Common techniques include residual-based error estimation, where the residuals of the finite element equations are used to estimate the local error, and recovery-based methods, which involve constructing a higher-order approximation of the solution and assessing the difference between this approximation and the finite element solution.
By identifying elements with high estimated errors, these methods provide a targeted approach to mesh refinement, enhancing the overall efficiency and accuracy of the simulation.
