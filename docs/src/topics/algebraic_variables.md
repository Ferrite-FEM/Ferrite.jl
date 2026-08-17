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
it enters the weak form. This is declared with a *coupling descriptor*, which is used to
allocate the corresponding test/trial blocks in the sparsity pattern. The weak-form
contribution itself is still assembled by user code.

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
augmented by the algebraic dofs. There is no dedicated data structure for this: since the
algebraic dofs are global and the same for every entity, the augmented dof vector is an
ordinary `Vector{Int}` where the constant tail from [`algebraic_dofs`](@ref) is written
once, outside the loop, and only the cell dofs are refreshed per entity. The assembler
only requires that the entries of the dof vector match the rows/columns of `Ke`, so the
local placement of the algebraic dofs is a choice made by the assembly code. Like
`CellValues`, [`AlgebraicValues`](@ref) is constructed once and passed to the element
routine:

```julia
σ̄values = AlgebraicValues(σ̄var)

function element_routine!(Ke, fe, ae, range_σ, cellvalues, σ̄values)
    σ̄ = algebraic_value(σ̄values, ae, range_σ)

    for (iσ, I) in pairs(range_σ)
        Eᵢ = algebraic_basis_value(σ̄values, iσ)
        # assemble the weak form at local index I using σ̄ and Eᵢ
    end
    return
end

function assemble_volume!(assembler, dh, ch, cellvalues, σ̄values, a)
    n = ndofs_per_cell(dh)
    nσ = getnbasefunctions(σ̄values)
    dofs = Vector{Int}(undef, n + nσ)
    dofs[(n + 1):end] .= algebraic_dofs(dh, :σ̄) # constant tail, written once
    range_σ = (n + 1):(n + nσ)                   # local placement of the σ̄ dofs
    Ke = zeros(eltype(a), n + nσ, n + nσ)
    fe = zeros(eltype(a), n + nσ)

    for cell in CellIterator(dh)
        copyto!(dofs, celldofs(cell)) # refresh the first n entries
        ae = a[dofs]
        fill!(Ke, 0)
        fill!(fe, 0)
        reinit!(cellvalues, cell)
        element_routine!(Ke, fe, ae, range_σ, cellvalues, σ̄values)
        apply_assemble!(assembler, ch, dofs, Ke, fe)
    end
    return
end
```

For an algebraic-only term the "local" dofs are simply the global algebraic dofs:

```julia
dofs = vcat(algebraic_dofs(dh, :p0), algebraic_dofs(dh, :z))
range_p0 = 1:1
range_z = 2:3
Ke = zeros(length(dofs), length(dofs))
fe = zeros(length(dofs))
# assemble the 0D contribution
assemble!(assembler, dofs, Ke, fe)
```

Since the augmented dof vector is built by user code, it must list the algebraic dofs of
the variables that the element routine writes to; forgetting a variable (or the
`algebraic_couplings` keyword during matrix allocation) surfaces as a
missing-sparsity-entry error on the first `assemble!`. After [`renumber!`](@ref), rebuild
the tail from the re-queried `algebraic_dofs`.

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
block with a Schur complement, or a block preconditioner), note that algebraic dofs are
always numbered after all spatial dofs, so a two-block split
`[spatial dofs | algebraic dofs]` (e.g. `[u, p | λ]` for Stokes with a multiplier) works
without renumbering, optionally combined with a [`BlockSparsityPattern`](@ref); the
[stress-driven homogenization tutorial](@ref tutorial-stress-driven-homogenization) shows
this. For per-field blocks (e.g. `[u | p | λ]`), renumber with
[`DofOrder.FieldWise`](@ref), which sorts the spatial fields into blocks and keeps each
algebraic variable as a trailing block.

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
