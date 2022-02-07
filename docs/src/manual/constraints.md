```@meta
DocTestSetup = :(using Ferrite)
```

# Constraints

PDEs can in general be subjected to a number of constraints, 

```math
g_I(\boldsymbol{a}) = 0, \quad I = 1 \text{ to } n_c
```

where ``g`` are (non-linear) constraint equations, $\boldsymbol{a}$ is a vector of the
degrees of freedom, and ``n_c`` is the number of constraints. There are many ways to
enforce these constraints, e.g. penalty methods and Lagrange multiplier methods. A special
case is when the constraint equations are affine/linear, in which they can be enforced in a
special way. This is explained below.

## Affine constraints

Affine (or linear) constraints can be written as

```math
a_i = \sum_j C_{ij} a_j + g_i
```

or in matrix form

```math
\boldsymbol{a}_c = \boldsymbol{C} \boldsymbol{a}_f + \boldsymbol{g}
```

where ``\boldsymbol{a}_c`` is a vector of the constrained dofs, ``\boldsymbol{a}_f`` is a
vector of free dofs, ``\boldsymbol{g}`` contains possible inhomogeneities, and
``\boldsymbol{C}`` is matrix defining the connectivity between constrained and free dofs.

...
