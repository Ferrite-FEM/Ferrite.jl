```@meta
DocTestSetup = :(using Ferrite)
```

# Constraints

PDEs can in general be subjected to a number of constraints, 

```math
g_I(\boldsymbol{a}) = 0, \quad I = 1 \text{ to } n_c
```

where `g` are (non-linear) constraint equations, $\boldsymbol{a}$ is a vector of the
degrees of freedom, and ``n_c`` is the number of constraints. There are many ways to
enforce these constraints, e.g. penalty methods and Lagrange multiplier methods. A special
case is when the constraint equations are affine/linear, explained below. 

## Affine constraints

Affine (or linear) constraints can be written as

```math
a_1 =  5a_2 + 3a_3 + 1 \\
a_4 =  2a_3 + 6a_5 \\
\dots
```

where $a_1$, $a_2$ etc. are system degrees of freedom. In Ferrite, we can account for 
such constraint using the `ConstraintHandler`:

```julia
ch = ConstraintHandler(dh)
lc1 = AffineConstraint(1, [2 => 5.0, 3 => 3.0], 1)
lc2 = AffineConstraint(1, [3 => 2.0, 5 => 6.0], 0)
add!(ch, lc1)
add!(ch, lc2)
```

When creating the sparsity pattern for the stiffness matrix, it is important to also include the `ConstraintHandler` 
as an argument because the affine constraints will affect the sparsity pattern:

```julia
K = create_sparsity_pattern(dh, ch)
```

When solving the system, we account for the affine constraints in the same way as we account for 
`Dirichlet` boundary conditions; by first calling `apply!(K, f, ch)`. This will condense `K` and `f` inplace (i.e
no new matrix will be created). Note however that we must also call `apply!` on the solution vector after 
solving the system to enforce the affine constraints (calling `apply!` after solving the system is not needed 
when we only have `Dirichlet` boundary conditions)

```julia
# ...
# Assemble K and f...

apply!(K, f, ch)
a = K\f
apply!(a, ch) # enforces affine constraints

```


...
