# [Algebraic variables](@id topic-algebraic-variables)

Some variational problems contain unknowns that live in a finite-dimensional space rather
than a function space over the physical domain. Examples include a scalar Lagrange
multiplier, the state of a coupled 0D model, and a macroscopic stress or strain in
computational homogenization.

Ferrite represents these unknowns with [`AlgebraicVariable`](@ref). An algebraic variable
is added to the `DofHandler` by name and receives globally numbered dofs, but has no mesh
domain and never appears in `celldofs`. Its coupling to spatial fields is declared
explicitly with a coupling descriptor.

## Declaring algebraic variables

Algebraic variables are added before `close!`, with a value shape and, optionally, a
selection of active components. The variables declared here serve as the running example
for the rest of this page:

```julia
dh = DofHandler(grid)
add!(dh, :u, ip_u)

p0var = AlgebraicVariable()
σ̄var = AlgebraicVariable{SymmetricTensor{2, 2}}()
add!(dh, :p0, p0var)                                      # one scalar dof
add!(dh, :z, AlgebraicVariable{Vec{3}}())                 # three dofs
add!(dh, :σ̄, σ̄var)                                      # three dofs
add!(dh, :σ̄₁, AlgebraicVariable{SymmetricTensor{2, 2}}(
    active_components = ((1, 1), (2, 2)),                 # two dofs
))
close!(dh)
```

The value shape may be scalar, `Vec{dim}`, or a second or fourth order `Tensor` or
`SymmetricTensor` (with `dim` in `1:3`). Only active components receive dofs, and spatial
fields and algebraic variables share one name namespace.

Use [`algebraic_dofs`](@ref) to query the global dof numbers. Renumbering updates these
numbers, so query them after the final call to [`renumber!`](@ref).

```julia
algebraic_dofs(dh, :σ̄₁)
```

## Values

[`algebraic_value`](@ref) reconstructs the typed value from a solution vector:

```julia
σ̄ = algebraic_value(dh, a, :σ̄₁) # ::SymmetricTensor{2, 2, Float64}
```

Inactive components are zero in the reconstructed value. When they represent known,
prescribed quantities, the total value is the prescribed part plus the reconstruction:

```julia
σ = σ_prescribed + algebraic_value(dh, a, :σ̄₁)
```

## Coupling descriptors

Since an algebraic variable has no mesh support, Ferrite cannot derive from the mesh where
it enters the weak form. This is declared with a *coupling descriptor*, which is used for
two things: to allocate the corresponding test/trial blocks in the sparsity pattern, and
to build the augmented local dof layouts during assembly. The weak-form contribution
itself is still assembled by user code.

The descriptor type selects the entities over which the terms are integrated:

 - [`CellCoupling`](@ref) for terms integrated over cells;
 - [`FacetCoupling`](@ref) for terms integrated over facets; and
 - [`AlgebraicCoupling`](@ref) for terms involving only algebraic variables.

The `algebraic_coupling` keyword lists the coupled variable pairs: `:u => :p0` declares a
directional coupling (test dofs of `:u` may couple to trial dofs of `:p0`), and
`(:u, :p0)` declares both directions. Continuing the example, `:σ̄` is a multiplier acting
in every cell, `:p0` one acting on part of the boundary, and `:z` the state of a 0D model
driven by `:p0`:

```julia
couplings = (
    volume = CellCoupling(
        dh, 1:getncells(grid);
        algebraic_coupling = ((:u, :σ̄),),
    ),
    boundary = FacetCoupling(
        dh, controlled_boundary;
        algebraic_coupling = ((:u, :p0),),
    ),
    zero_d = AlgebraicCoupling(
        dh;
        algebraic_coupling = ((:p0, :z), (:z, :z)),
    ),
)
```

Diagonal matrix entries are always allocated, but off-diagonal entries between the
components of a single algebraic variable are not implicit: if a term couples the
variable to itself, declare the self-pair, like `(:z, :z)` above for the dense Jacobian
of the 0D model. (An energy term in a homogenization problem is another example.)

Pass a descriptor or collection of descriptors to matrix allocation (the
[`add_sparsity_entries!`](@ref) function accepts the same keyword when building a
sparsity pattern manually, e.g. a [`BlockSparsityPattern`](@ref)):

```julia
K = allocate_matrix(dh, ch; algebraic_couplings = couplings)
```

The descriptor entries are added on top of the ordinary cell sparsity. Descriptors may
overlap: their sparsity entries are unioned, and their numerical contributions add during
assembly.

## Assembly

Assembly over a descriptor's entities works like ordinary assembly, with the local system
augmented by the algebraic dofs: [`local_dofs!`](@ref) fills a [`LocalDofLayout`](@ref)
with the cell dofs followed by the descriptor's algebraic dofs, and [`dof_range`](@ref)
gives the local range of each variable. Like `CellValues`, [`AlgebraicValues`](@ref) is
constructed once and passed to the element routine:

```julia
σ̄values = AlgebraicValues(σ̄var)

function element_routine!(Ke, fe, ae, layout, cellvalues, σ̄values)
    range_σ = dof_range(layout, :σ̄)
    σ̄ = algebraic_value(σ̄values, ae, range_σ)

    for (iσ, I) in pairs(range_σ)
        Eᵢ = algebraic_basis_value(σ̄values, iσ)
        # assemble the weak form at local index I using σ̄ and Eᵢ
    end
    return
end

function assemble_volume!(assembler, dh, ch, coupling, cellvalues, σ̄values, a)
    nσ = getnbasefunctions(σ̄values)
    nlocal = ndofs_per_cell(dh) + nσ
    Ke = zeros(eltype(a), nlocal, nlocal)
    fe = zeros(eltype(a), nlocal)
    layout = LocalDofLayout()

    for cell in CellIterator(dh)
        local_dofs!(layout, cell, coupling)
        ae = a[layout]
        fill!(Ke, 0)
        fill!(fe, 0)
        reinit!(cellvalues, cell)
        element_routine!(Ke, fe, ae, layout, cellvalues, σ̄values)
        apply_assemble!(assembler, ch, layout, Ke, fe)
    end
    return
end
```

For an algebraic-only term:

```julia
layout = local_dofs!(LocalDofLayout(), couplings.zero_d)
range_p0 = dof_range(layout, :p0)
range_z = dof_range(layout, :z)
Ke = zeros(length(layout), length(layout))
fe = zeros(length(layout))
# assemble the 0D contribution
assemble!(assembler, layout, Ke, fe)
```

Note that `dof_range(layout, :p0)` is a range of *local* indices into the augmented
system, not the global dof numbers (those are given by `algebraic_dofs(dh, :p0)`).

## Threading and matrix structure

Algebraic dofs are shared between all entities in their coupling set, so cell coloring
does not make those writes independent. Use atomic assembly when processing the entities
concurrently:

```julia
assembler = start_assemble(K, f; atomic = true)
```

A variable coupled to a field over the whole domain gives the matrix a few dense rows and
columns. This is inherent to global coupling and nothing to fix at the sparsity level.
If the solver should exploit the structure instead (e.g. eliminating the small algebraic
block with a Schur complement, or a block preconditioner), renumber with
[`DofOrder.FieldWise`](@ref) to group the algebraic dofs into their own block, optionally
combined with a [`BlockSparsityPattern`](@ref); the
[stress-driven homogenization tutorial](@ref tutorial-stress-driven-homogenization) shows
this.

## Scope

Algebraic variables have no interpolation or spatial evaluation. Operations that require
a spatial field, such as Dirichlet conditions, point evaluation, projection, and VTK
export, do not accept algebraic names. To constrain an algebraic dof, use an
[`AffineConstraint`](@ref) with [`algebraic_dofs`](@ref).

They are intended for a small number of global unknowns. A multiplier field with one
unknown per node or facet has spatially local support and should instead be represented by
an appropriate spatial or trace field.

See the [stress-driven homogenization](@ref tutorial-stress-driven-homogenization) and
[Stokes flow](@ref tutorial-stokes-flow) tutorials for complete examples.
