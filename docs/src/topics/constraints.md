```@meta
DocTestSetup = :(using Ferrite)
```

# Affine constraints

In FEM, affine constraints (sometimes called linear constraints) are commonly used to couple
different degrees of freedom (DoFs). These constraints arise in cases such as [periodic
boundary conditions](@ref "Periodic boundary conditions"), where the solution (and thus the
DoFs) on one face should behave periodically with the solution on the opposing face. The
constraint then take the following form, with DoF ``a_1`` mirroring DoF ``a_2``:

```math
a_1 = a_2,
```

Another common case are "hanging node" constraints, where the value at the hanging node
(``a_1``) should be constrained to a weighted combination of the two adjacent nodes (``a_2``
and ``a_3``), i.e.

```math
a_1 = 0.5a_2 + 0.5a_3.
```

Furthermore, affine constraints can also been viewed as a generalization of Dirichlet
boundary conditions of the form ``a_1 = c`` for some prescribed value ``c``. However,
Dirichlet BCs are mathematically and implementation-wise much easier to handle.

To enforce affine constraints, the original linear system needs to be modified. To explain
this process, consider a problem with six DoFs where we have the following linear system of
equations (obtained from a standard finite element procedure):

```math
\boldsymbol{K} \boldsymbol{a} = \boldsymbol{f},
```

which is subjected to the following constraints:

```math
a_1 = 5a_2 + 3a_3 + 1 \\
a_4 = 2a_3 + 6a_5
```

To incorporate these constraints into the linear system, we first collect the coefficients
into a constraint matrix, ``\boldsymbol{C}``, and the inhomogeneities into a vector,
``\boldsymbol{g}``:

```math
\boldsymbol{a} = \boldsymbol{C} \boldsymbol{a}_f + \boldsymbol{g}
```

where

```math
C =
\begin{bmatrix}
5 & 3 & 0 & 0 \\
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 2 & 6 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}, \quad
g =
\begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}
```

and where ``\boldsymbol{a}_f = [a_2, a_3, a_5, a_6]`` contains the "free" DoFs, and ``a_1``
and ``a_4`` are dependent on the others. Next, the above equation is inserted into the
original system, and if we pre-multiply with ``\boldsymbol{C}^T``, we get a reduced system
of equations:

```math
\hat{\boldsymbol{K}} \boldsymbol{a}_f = \hat{\boldsymbol{f}}, \quad \hat{\boldsymbol{K}} = \boldsymbol{C}^T \boldsymbol{K} \boldsymbol{C}, \quad \hat{\boldsymbol{f}} = \boldsymbol{C}^T(\boldsymbol{f} - \boldsymbol{K}\boldsymbol{g})
```

The reduced system of equations can used to solve for ``\boldsymbol{a}_f``, which can then
be used to calculate the dependent DoFs. Ferrite has functionaliy for setting up the
``\hat{\boldsymbol{K}}`` and ``\hat{\boldsymbol{f}}`` in an efficient way.

!!! note "Limitations"
    Ferrite currently cannot untangle constraints where a DoF is both a *main* and a *constrained*
    DoF. For example, if we have two affine constraints such as:
    ```math
    a_1 = 2a_2 + 4 \\
    a_2 = 3a_3 + 1
    ```
    Ferrite will not be able to resolve this situation because `` a_2 `` is both a main
    and a constrained DoF in different constraints.

### Affine Constraints in Ferrite

To explain how affine constraints are handled in Ferrite, we will use the same example as
above. The constraint equations can be constructed with `Ferrite.AffineConstraints` and
added to the `ConstraintHandler`:

```julia
ch = ConstraintHandler(dh)

lc1 = Ferrite.AffineConstraint(1, [2 => 5.0, 3 => 3.0], 1.0)
lc2 = Ferrite.AffineConstraint(4, [3 => 2.0, 5 => 6.0], 0.0)

add!(ch, lc1)
add!(ch, lc2)
```

Affine constraints will impact the sparsity pattern of the matrix, and as such, it is
important to also include the `ConstraintHandler` as an argument when creating the sparsity
pattern:

```julia
K = allocate_matrix(dh, ch)
```

To solve the system, we could create ``\boldsymbol{C}`` and ``\boldsymbol{g}`` with the function

```julia
C, g = Ferrite.create_constraint_matrix(ch)
```

and condense ``\boldsymbol{K}`` and ``\boldsymbol{f}`` as described above. However, this is
in general inefficient, since the triple matrix product `C'*K*C` is expensive and will
allocate extra memory. Instead, Ferrite provides a much more efficient way of doing this.
In fact, the affine constraints can be accounted for in the same way as `Dirichlet` boundary
conditions; first call `apply!(K, f, ch)`, which will condense `K` and `f` inplace (i.e no
new matrix will be created), and secondly call `apply!` on the solution vector after solving
the system to enforce the affine constraints:

```julia
# Assemble K and f...
K, f = assemble_system(...)
# Apply constraints in place in K and f
apply!(K, f, ch)
# Solve linear system
a = K \ f
# Compute dependent values to make sure constraints are fulfilled
apply!(a, ch)
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
