# [Algebraic variables](@id topic-algebraic-variables)

Some variational problems contain unknowns that live in a finite dimensional space such as
``\mathbb{R}``, ``\mathbb{R}^n``, or the space of symmetric tensors, rather than in a
function space over the physical domain. Typical examples are

 - a scalar Lagrange multiplier enforcing an integral constraint (e.g. a mean value
   constraint),
 - the state of a coupled 0D model (e.g. a lumped circulation model coupled to a 3D
   problem through a boundary),
 - an unknown macroscopic stress or strain in computational homogenization.

Ferrite represents such unknowns with [`AlgebraicVariable`](@ref). An algebraic variable
is added to the `DofHandler` by name, receives globally numbered degrees of freedom
(after all spatial dofs), but has *no* mesh domain: its dofs never appear in `celldofs`.
Where the variable couples to spatial fields is instead declared explicitly with
*coupling descriptors*, and assembly of the coupled terms uses an *augmented local dof
layout* through Ferrite's ordinary square local-matrix assembly path.

The final model has three independent pieces:

```
AlgebraicVariable   what typed value and coordinate subspace the dofs represent
Coupling descriptor on which entities and between which test/trial variables entries may occur
User assembly      the actual weak-form contribution evaluated on those entities
```

!!! note "Algebraic variables are not spatially constant fields"
    A field that is *constant over a domain* (a "spatial constant") is a spatial
    approximation space: it is attached to a domain, has a (constant) basis function with
    spatial support, and naturally couples to everything on that domain. An algebraic
    variable is domain-free: it has no basis function over the mesh and receives support
    only through explicit coupling descriptors. This distinction is what allows e.g. a 0D
    variable to couple only to the cells adjacent to a boundary, keeping the matrix
    sparse.

## Declaring algebraic variables

Algebraic variables are added to the `DofHandler` before `close!`, with a declared value
shape and, optionally, a selection of *active components*:

```julia
dh = DofHandler(grid)
add!(dh, :u, ip_u) # ordinary spatial field

add!(dh, :p0, AlgebraicVariable())                        # one scalar unknown
add!(dh, :z, AlgebraicVariable{Vec{3}}())                 # three unknowns
add!(dh, :σ̄, AlgebraicVariable{SymmetricTensor{2, 2}}())  # three unknowns
add!(dh, :σ̄₁, AlgebraicVariable{SymmetricTensor{2, 2}}(
    active_components = ((1, 1), (2, 2)),                 # two unknowns
))
close!(dh)
```

Only active components receive dofs; a partially active variable declares a coordinate
subspace of the value type, e.g. macroscopic stresses where some components are prescribed
data. Spatial fields and algebraic variables share one name namespace.

After `close!` the variables can be queried:

```julia
getalgebraicvariablenames(dh) # [:p0, :z, :σ̄, :σ̄₁]
algebraic_variable(dh, :σ̄₁)   # the AlgebraicVariable descriptor
algebraic_dofs(dh, :σ̄₁)       # current global dof numbers, in active-component order
active_components(algebraic_variable(dh, :σ̄₁)) # ((1, 1), (2, 2))
```

`algebraic_dofs` returns the dof numbers valid at the time of the call: renumbering with
[`renumber!`](@ref) updates the stored numbers (each algebraic variable is one block for
`DofOrder.FieldWise`, and its active components are individual components for
`DofOrder.ComponentWise`), but the returned vector is a copy and the numbers are not
guaranteed to be contiguous after renumbering.

## Values and basis directions

The typed value of a variable is reconstructed from a solution vector with
[`algebraic_value`](@ref):

```julia
σ̄ = algebraic_value(dh, a, :σ̄₁) # ::SymmetricTensor{2, 2, Float64}
```

Inactive components are reconstructed as typed zeros; prescribed values for them are
problem data that user code adds explicitly:

```julia
σ = σ_prescribed + algebraic_value(dh, a, :σ̄₁)
```

This name-based method is convenient for postprocessing, but it looks the variable up in
an untyped registry and is therefore not type stable. Inside assembly kernels the
variable is instead queried on an [`AlgebraicValues`](@ref) — the algebraic counterpart
of `CellValues`. Just like a `CellValues` is constructed from the interpolation of a
field, an `AlgebraicValues` is constructed from the `AlgebraicVariable` during setup,
and both are passed into the kernel:

```julia
cv = CellValues(qr, ip_u)      # ip_u is the interpolation added as the field :u
av = AlgebraicValues(σ̄₁var)    # σ̄₁var is the AlgebraicVariable added as :σ̄₁
```

(When the variable binding is not at hand, construct from the registry lookup:
`AlgebraicValues(algebraic_variable(dh, :σ̄₁))`.)

Since the variable has no mesh domain there is no quadrature, geometry, or `reinit!`;
the `AlgebraicValues` carries the constant basis directions and makes all queries type
stable. The value is reconstructed from a coefficient vector, typically the local
unknowns of the augmented local system with the variable's local `dof_range` (see
[`local_dofs`](@ref) below):

```julia
σ̄ = algebraic_value(av, ae, dof_range(layout, :σ̄₁))
```

The scalar type follows the input, so this also works when the local unknowns carry dual
numbers for automatic differentiation of the kernel.

For linearization, [`algebraic_basis_value`](@ref) returns the constant direction
associated with each active dof — the derivative of the reconstructed value with respect
to that coefficient. These directions take the place of the variable's test/trial
functions in the weak form, but they are not shape functions: they have no spatial
argument, gradient, or quadrature:

```julia
E_i = algebraic_basis_value(av, i)
```

## Coupling descriptors

Where an algebraic variable couples to spatial fields is *not* inferred from the mesh; it
is declared with one of three structural descriptors:

 - [`CellCoupling`](@ref)`(dh, cells; algebraic_coupling)` for terms integrated over
   cells;
 - [`FacetCoupling`](@ref)`(dh, facets; algebraic_coupling)` for terms integrated over
   facets; and
 - [`AlgebraicCoupling`](@ref)`(dh; algebraic_coupling)` for terms involving only
   algebraic variables.

A descriptor carries only structural metadata: the entity set and the coupled test/trial
blocks. It does not carry coefficients, quadrature, or the integrand — the weak-form
kernel remains user code. The `algebraic_coupling` specification is a collection of
entries, each involving at least one algebraic variable: `:u => :p0` declares a
directional coupling (test dofs of `:u` may couple to trial dofs of `:p0`), and the tuple
`(:u, :p0)` declares both directions at once.

```julia
couplings = (
    volume_control = FacetCoupling(
        dh, controlled_boundary;
        algebraic_coupling = ((:u, :p0),),      # u-p0 and p0-u blocks
    ),
    zero_d_problem = AlgebraicCoupling(
        dh;
        algebraic_coupling = ((:p0, :z), (:z, :z)),
    ),
)
```

For component-level control the coupled blocks can instead be given as a Boolean matrix,
where rows are test variables and columns trial variables (possibly asymmetric), together
with a `fields` tuple that orders the participating names, see [`CellCoupling`](@ref).

Matrix allocation accepts a descriptor, or any iterable of descriptors (a named tuple is
recommended, so that assembly can refer to `couplings.volume_control` by name), through
the `algebraic_couplings` keyword and unions their entries with the ordinary cell pattern:

```julia
K = allocate_matrix(dh, ch; algebraic_couplings = couplings)
```

Coupling entries are inserted after the ordinary cell (and interface) entries and before
the constraint entries, so affine-constraint expansion sees all structural entries.

!!! note "Couplings are additive"
    The ordinary cell sparsity and the existing `coupling` keyword keep their meaning;
    `algebraic_couplings` only *adds* structural entries. Consequently, a descriptor that
    does not declare e.g. a `u`-`u` block only means "this descriptor adds no `u`-`u`
    entries" — it does not remove the entries the ordinary cell pass provides.
    Overlapping descriptors union their sparsity, and overlapping numerical contributions
    add during assembly.

!!! note "Facet couplings use all dofs of the adjacent cell"
    A facet integral may evaluate cell gradients on the facet (the gradient of a shape
    function associated with a non-facet node can be nonzero there), so the local system
    of a `FacetCoupling` term contains *all* dofs of the adjacent cell, not only the dofs
    located on the facet. Only cells adjacent to the selected facets couple to the
    algebraic variables.

## Assembly with augmented local layouts

While matrix allocation loops uniformly over all descriptors, user assembly normally
selects one descriptor by name per weak-form term, since each term has its own kernel,
quadrature, and iterator. Assembly uses the ordinary `CellIterator`/`FacetIterator`
together with [`local_dofs`](@ref), which returns a [`LocalDofLayout`](@ref): the ordinary
cell dofs followed by the descriptor's algebraic dofs, with named ranges. The square local
matrix over this layout is passed to the existing [`assemble!`](@ref) (or
[`apply_assemble!`](@ref)) methods:

```julia
descriptor = couplings.volume_control
av = AlgebraicValues(algebraic_variable(dh, :p0))

nlocal = ndofs_per_cell(dh) + ndofs(av)
Ke = zeros(nlocal, nlocal) # hoisted out of the loop
fe = zeros(nlocal)

for facet in FacetIterator(dh, controlled_boundary)
    layout = local_dofs(facet, descriptor)
    range_u = dof_range(layout, :u)
    range_p0 = dof_range(layout, :p0)
    fill!(Ke, 0)
    fill!(fe, 0)
    reinit!(facetvalues, facet)
    # ... evaluate the weak form into Ke[range_u, range_p0] etc ...
    assemble!(assembler, layout, Ke, fe)
end
```

and for an algebraic-only term:

```julia
descriptor = couplings.zero_d_problem
layout = local_dofs(descriptor)
range_p0 = dof_range(layout, :p0)
range_z = dof_range(layout, :z)
Ke = zeros(length(layout), length(layout))
fe = zeros(length(layout))
# ... evaluate the 0D tangent and residual ...
assemble!(assembler, layout, Ke, fe)
```

Note the three distinct questions answered by three distinct APIs:
`algebraic_dofs(dh, :p0)` returns current *global* dof numbers,
`algebraic_value` returns the represented *value* (from the global vector by name, or
type stably from local coefficients through an `AlgebraicValues`), and
`dof_range(layout, :p0)` returns a *local* range in one augmented local system (there is
deliberately no `dof_range(dh, :p0)`, since the augmented local range depends on the
descriptor). Unlike `dof_range(::InterfaceCache, field)`, which returns one range per
interface side, `dof_range(::LocalDofLayout, name)` returns a single range.

For a nonlinear problem the kernel is a residual evaluated at the current iterate: gather
the augmented local unknowns with `ae = a[layout]` and reconstruct the value inside the
routine with `algebraic_value(av, ae, dof_range(layout, :p0))`. Since the reconstruction
follows the scalar type of `ae`, differentiating the routine with respect to `ae` (e.g.
with ForwardDiff) yields the full augmented element matrix, including the coupling
blocks.

`local_dofs` allocates a fresh layout for every entity. Hot assembly loops can instead
hoist an empty layout out of the loop and update it in place with [`local_dofs!`](@ref):

```julia
layout = LocalDofLayout()
for facet in FacetIterator(dh, controlled_boundary)
    local_dofs!(layout, facet, descriptor)
    # ...
end
```

## Example: stress-driven computational homogenization

In stress-driven (Neumann-type) homogenization of an RVE, the macroscopic stress
``\bar{\sigma}`` is an unknown symmetric tensor acting as the Lagrange multiplier that
enforces the prescribed average strain ``\bar{\varepsilon}``: find ``u`` and
``\bar{\sigma}`` such that

```math
\int_\Omega \varepsilon(\delta u) : \mathsf{C} : \varepsilon(u) \, d\Omega
- \int_\Omega \varepsilon(\delta u) : \bar{\sigma} \, d\Omega = 0
\qquad \forall\, \delta u,
```
```math
- \int_\Omega \delta\bar{\sigma} : \varepsilon(u) \, d\Omega
= - |\Omega|\, \delta\bar{\sigma} : \bar{\varepsilon}
\qquad \forall\, \delta\bar{\sigma}.
```

With an algebraic variable for ``\bar{\sigma}`` and a `CellCoupling` over the RVE this
assembles as one augmented square local system per cell:

```julia
σ̄var = AlgebraicVariable{SymmetricTensor{2, 2}}()
dh = DofHandler(grid)
add!(dh, :u, ip)
add!(dh, :σ̄, σ̄var)
close!(dh)

descriptor = CellCoupling(
    dh, 1:getncells(grid);
    # u-σ̄ and σ̄-u blocks; no σ̄-σ̄ entries needed (the diagonal is always stored)
    algebraic_coupling = ((:u, :σ̄),),
)
K = allocate_matrix(dh, ch; algebraic_couplings = (rve = descriptor,))
f = zeros(ndofs(dh))

# The AlgebraicValues is created next to the CellValues and passed into the assembly
# function together with the other reusable data.
av = AlgebraicValues(σ̄var)

function assemble_rve!(assembler, ch, dh, descriptor, cellvalues, av, C, ε̄)
    nlocal = ndofs_per_cell(dh) + ndofs(av)
    Ke = zeros(nlocal, nlocal)
    fe = zeros(nlocal)
    for cell in CellIterator(dh)
        layout = local_dofs(cell, descriptor)
        range_u = dof_range(layout, :u)
        range_σ = dof_range(layout, :σ̄)
        fill!(Ke, 0); fill!(fe, 0)
        reinit!(cellvalues, cell)
        for qp in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, qp)
            for (i, I) in pairs(range_u)
                εi = shape_symmetric_gradient(cellvalues, qp, i)
                for (j, J) in pairs(range_u)
                    εj = shape_symmetric_gradient(cellvalues, qp, j)
                    Ke[I, J] += (εi ⊡ C ⊡ εj) * dΩ
                end
                for (k, S) in pairs(range_σ)
                    Ek = algebraic_basis_value(av, k)
                    Ke[I, S] -= (εi ⊡ Ek) * dΩ
                    Ke[S, I] -= (Ek ⊡ εi) * dΩ
                end
            end
            for (k, S) in pairs(range_σ)
                fe[S] -= (algebraic_basis_value(av, k) ⊡ ε̄) * dΩ
            end
        end
        apply_assemble!(assembler, ch, layout, Ke, fe)
    end
    return
end

assembler = start_assemble(K, f)
assemble_rve!(assembler, ch, dh, descriptor, cellvalues, av, C, ε̄)
K, f = finish_assemble(assembler)
a = K \ f
apply!(a, ch)

σ̄ = algebraic_value(dh, a, :σ̄) # the macroscopic stress
```

For a homogeneous material this recovers ``\bar{\sigma} = \mathsf{C} : \bar{\varepsilon}``
and ``u = \bar{\varepsilon} \cdot x`` exactly (up to the rigid body modes removed by the
constraint handler `ch`).

## Threaded assembly

The dofs of an algebraic variable are shared between all entities of its coupling set, so
cell coloring cannot make those writes independent. The supported threaded path is the
atomic assembler:

```julia
assembler = start_assemble(K, f; atomic = true)
```

where each task owns its assembler and local buffers. Atomic assembly supports
`SparseMatrixCSC`, `Symmetric`-wrapped `SparseMatrixCSC`, `SparseMatrixCSR`, and the
BlockArrays assembler with `Float32`/`Float64` values.

## Dense rows and block structure

A `CellCoupling` over the whole domain creates dense rows and columns for the algebraic
dofs. Field-wise renumbering ([`DofOrder.FieldWise`](@ref)) together with
[`BlockSparsityPattern`](@ref) exposes this structure as blocks, which applications can
exploit e.g. with a Schur complement on the (small) algebraic block; Ferrite provides the
blocking, not the solver. Note that the Metis renumbering extension builds its dof graph
from the cell dofs only and is unaware of coupling descriptors, so algebraic dofs are
ordered as isolated vertices.

## What algebraic variables are not

Algebraic variables have no interpolation, quadrature, or spatial evaluation: APIs that
require a spatial field — name-based `Dirichlet`, `PeriodicDirichlet`,
`ProjectedDirichlet`, `apply_analytical!`, `evaluate_at_grid_nodes`, point evaluation,
projection, and VTK export — reject algebraic names with descriptive errors (and continue
to work for the spatial fields of the same handler). Whole-solution export
(`write_solution`) writes the spatial fields and omits algebraic variables. To constrain
an algebraic dof, use the dof-number based [`AffineConstraint`](@ref) with
`algebraic_dofs(dh, name)`.

They are also meant for a *small number* of unknowns — a handful of named variables with
a few dofs each, as in the examples above. A field of multipliers with one unknown per
node or facet of a surface (e.g. the contact pressure in mortar/contact methods) is a
*trace field*: it has spatially local support and a choice of interpolation, and
representing it as many algebraic variables would require one variable and one coupling
descriptor per entity to retain sparsity. Such interface fields are outside what
algebraic variables are designed for.
