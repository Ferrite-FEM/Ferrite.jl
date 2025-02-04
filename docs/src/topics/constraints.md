```@meta
DocTestSetup = :(using Ferrite)
```

# Affine Constraints  

In FEM, it is common to use affine (or linear) constraints where one degree of freedom (DOF) depends on other DOFs. These types of constraints appear, for example, in [periodic boundary conditions](@ref "Periodic boundary conditions"), where the DOF (or node) on one face should deform periodically with a corresponding node on the opposing face:  

```math
a_1 = a_2
```

or in hanging nodes, where the value at the hanging node (or DOF) should be a weighted combination of the adjacent nodes:  

```math
a_1 = 0.5a_3 + 0.5a_4.
```

In the above equations ``a_1, a_2, \dots`` represents the degree of freedom at node ``1, 2, \dots `` and so on. Furtheremore, note that Dirichlets BCs can be seen as a special case of affine constraints, e.g. ``a_1 = 2``, however they are mathematically and implementation wise much easier to handle.

Affine constraints are typically enforced by modifying the original linear system of equations. To explain this process, consider a problem with six DOFs where we have the following linear system of equations obtained from the finite element procedure: 

```math
\boldsymbol{K} \boldsymbol{a} = \boldsymbol{f}
```

and is subjected to the follwoing affine constraints:

```math
a_1 = 5a_2 + 3a_3 + 1 \\
a_4 = 2a_3 + 6a_5
```

To account for the these constraints when solving the linear system, we first collect them into a constraint matrix, ``\boldsymbol{C}`` and a vector containing the inhomogeneities, ``\boldsymbol{g}``, which relate the system DOFs to the free ones:  

```math
\boldsymbol{a} = \boldsymbol{C} \boldsymbol{a}_f + \boldsymbol{g}
```

with

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

where ``\boldsymbol{a}_f = [a_2, a_3, a_5, a_6 ]`` are the free DOFs, and ``a_1`` and ``a_4`` are dependent on the others. Next, Equation (1) is inserted into the original system, and if we pre-multiply with ``\boldsymbol{C}^T``, we get a reduced system of equations:  

```math
\hat{\boldsymbol{K}} \boldsymbol{a}_f = \hat{\boldsymbol{f}}, \quad \hat{\boldsymbol{K}} = \boldsymbol{C}^T \boldsymbol{K} \boldsymbol{C}, \quad \hat{\boldsymbol{f}} = \boldsymbol{C}^T(\boldsymbol{f} - \boldsymbol{K}\boldsymbol{g})
```

The reduced system of equations can used to solve for `` a_f ``, which can then be used to calculate the dependent DOFs. Ferrite has functionaliy for setting up the ``\hat{\boldsymbol{K}}`` and ``\hat{\boldsymbol{f}}`` in an efficient way.

!!! note "Limitation" 
    Ferrite currently cannot handle (untangle) constraints where a DOF is both **master** and **slave** DOF. For example, if we have two affine constraints such as: 
    ```math
    a_1 = 2a_2 + 4 \\
    a_2 = 3a_3 + 1
    ``` 
    Ferrite will not be able to resolve this situation because `` a_2 `` is both a master and a slave DOF in different constraints. 

### Affine Constraints in Ferrite  

To explain how affine constraints are handled in Ferrite, we will use the same example as above. The constraint equations can be constructed with `Ferrite.AffineConstraints` and added to the `ConstraintHandler`:  

```julia
ch = ConstraintHandler(dh)

lc1 = Ferrite.AffineConstraint(1, [2 => 5.0, 3 => 3.0], 1)
lc2 = Ferrite.AffineConstraint(4, [3 => 2.0, 5 => 6.0], 0)

add!(ch, lc1)
add!(ch, lc2)
```

Affine constraints will affect the sparsity pattern of the stiffness matrix, and as such, it is important to also include the `ConstraintHandler` as an argument when creating the sparsity pattern:  

```julia
K = allocate_matrix(dh, ch)
```

To solve the system, we could create ``\boldsymbol{C}`` and ``\boldsymbol{g}`` with the function 

```julia
C, g = Ferrite.create_constraint_matrix(ch)
```

and condense ``\boldsymbol{K}`` and ``\boldsymbol{f}`` as described above. However, this is in general inefficient, since the operation `C'*K*C` will allocate temporary matrices. Instead, Ferrite provides a much more efficient way of doing this. Infact, the affine constraints can be accounted for in the same way as
`Dirichlet` boundary conditions; first call `apply!(K, f, ch)` which will condense `K` and `f` 
inplace (i.e no new matrix will be created), and secondly call `apply!` on 
the solution vector after solving the system to enforce the affine constraints:

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
