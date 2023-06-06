```@meta
DocTestSetup = :(using Ferrite)
```

# Constraints

PDEs can in general be subjected to a number of constraints, 

```math
g_I(\underline{a}) = 0, \quad I = 1 \text{ to } n_c
```

where $g$ are (non-linear) constraint equations, $\underline{a}$ is a vector of the
degrees of freedom, and $n_c$ is the number of constraints. There are many ways to
enforce these constraints, e.g. penalty methods and Lagrange multiplier methods. 

## Affine constraints

Affine or linear constraints can be handled directly in Ferrite. Such constraints can typically
be expressed as:

```math
a_1 =  5a_2 + 3a_3 + 1 \\
a_4 =  2a_3 + 6a_5 \\
\dots
```

where $a_1$, $a_2$ etc. are system degrees of freedom. In Ferrite, we can account for such constraint using the `ConstraintHandler`:

```julia
ch = ConstraintHandler(dh)
lc1 = AffineConstraint(1, [2 => 5.0, 3 => 3.0], 1)
lc2 = AffineConstraint(1, [3 => 2.0, 5 => 6.0], 0)
add!(ch, lc1)
add!(ch, lc2)
```

Affine constraints will affect the sparsity pattern of the stiffness matrix, and as such, it is important to also include 
the `ConstraintHandler` as an argument when creating the sparsity pattern:

```julia
K = create_matrix(dh, ch)
```

### Solving linear problems
To solve the system ``\underline{\underline{K}}\underline{a}=\underline{f}``, account for affine constraints the same way as for 
`Dirichlet` boundary conditions; first call `apply!(K, f, ch)`. This will condense `K` and `f` inplace (i.e
no new matrix will be created). Note however that we must also call `apply!` on the solution vector after 
solving the system to enforce the affine constraints:

```julia
# ...
# Assemble K and f...

apply!(K, f, ch)
a = K\f
apply!(a, ch) # enforces affine constraints

```

### Solving nonlinear problems
It is important to check the residual **after** applying boundary conditions when 
solving nonlinear problems with affine constraints. 
`apply_zero!(K, r, ch)` modifies the residual entries for dofs that are involved 
in constraints to account for constraint forces. 
The following pseudo-code shows a typical pattern for solving a non-linear problem with Newton's method:
```julia
a = initial_guess(...)  # Make any initial guess for a here, e.g. `a=zeros(ndofs(dh))`
apply!(a, ch)           # Make the guess fulfill all constraints in `ch`
for iter in 1:maxiter
    doassemble!(K, r, ...)  # Assemble the residual, r, and stiffness, K=∂r/∂a.
    apply_zero!(K, r, ch)   # Modify `K` and `r` to account for the constraints. 
    check_convergence(r, ...) && break # Only check convergence after `apply_zero!(K, r, ch)`
    Δa = K \ r              # Calculate the (negative) update
    apply_zero!(Δa, ch)     # Change the constrained values in `Δa` such that `a-Δa`
                            # fulfills constraints if `a` did.
    a .-= Δa
end
```
