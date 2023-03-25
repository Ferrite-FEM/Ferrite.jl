#####################
## DoF renumbering ##
#####################

# Namespace for orderings for the purpose of only exporting the DofOrder name.
# Implementations and documentation below.
module DofOrder

    function _check_target_blocks(x)
        if !(isempty(x) || sort!(unique(x)) == 1:maximum(x))
            error("target blocks must be continuous and in the range 1:maxblock")
        end
        return x
    end

    # abstract type DofOrdering end

    struct FieldWise # <: DofOrdering
        target_blocks::Vector{Int}
        FieldWise(x=Int[]) = new(_check_target_blocks(x))
    end

    struct ComponentWise # <: DofOrdering
        target_blocks::Vector{Int}
        ComponentWise(x=Int[]) = new(_check_target_blocks(x))
    end

    """
        DofOrder.Ext{T}

    DoF permutation order from external package `T`. Currently supported extensions:
    - `DofOrder.Ext{Metis}`: Fill-reducing permutation from
      [Metis.jl](https://github.com/JuliaSparse/Metis.jl).
    """
    abstract type Ext{T} end
    function Ext{T}(args...; kwargs...) where T
        throw(ArgumentError("Unknown external order DofOrder.Ext{$T}. See documentation for `DofOrder.Ext` for details."))
    end

end # module DofOrder

"""
    renumber!(dh::AbstractDofHandler, order)
    renumber!(dh::AbstractDofHandler, ch::ConstraintHandler, order)

Renumber the degrees of freedom in the DofHandler and/or ConstraintHandler according to the
ordering `order`.

`order` can be given by one of the following options:
 - A permutation vector `perm::AbstractVector{Int}` such that dof `i` is renumbered to
   `perm[i]`.
 - [`DofOrder.FieldWise()`](@ref) for renumbering dofs field wise.
 - [`DofOrder.ComponentWise()`](@ref) for renumbering dofs component wise.
 - `DofOrder.Ext{T}` for "external" renumber permutations, see documentation for
   `DofOrder.Ext` for details.

!!! warning
    The dof numbering in the DofHandler and ConstraintHandler *must always be consistent*.
    It is therefore necessary to either renumber *before* creating the ConstraintHandler in
    the first place, or to renumber the DofHandler and the ConstraintHandler *together*.
"""
renumber!

# Entrypoints
renumber!(dh::AbstractDofHandler, order) = _renumber!(dh, nothing, order)
renumber!(dh::AbstractDofHandler, ch::ConstraintHandler, order) = _renumber!(dh, ch, order)


# Internal methods: a new order type `O` should implement either
# `compute_renumber_permutation(::DofHandler, ::ConstraintHandler, ::O)` returning the dof
# permutation or `_renumber!(::DofHandler, ::ConstraintHandler, ::O)`.

function _renumber!(dh::AbstractDofHandler, ch::Union{ConstraintHandler,Nothing}, order)
    @assert ch === nothing || ch.dh === dh
    perm = compute_renumber_permutation(dh, ch, order)
    @assert isperm(perm) && length(perm) == ndofs(dh)
    _renumber!(dh, perm)
    ch === nothing || _renumber!(ch, perm)
    return
end

function _renumber!(dh::Union{DofHandler, MixedDofHandler}, perm::AbstractVector{<:Integer})
    @assert isclosed(dh)
    for i in eachindex(dh.cell_dofs)
        dh.cell_dofs[i] = perm[dh.cell_dofs[i]]
    end
    return dh
end

function _renumber!(ch::ConstraintHandler, perm::AbstractVector{<:Integer})
    @assert isclosed(ch)
    # To renumber the ConstraintHandler we start by renumbering master dofs in
    # ch.dofcoefficients.
    for coeffs in ch.dofcoefficients
        coeffs === nothing && continue
        for (i, (k, v)) in pairs(coeffs)
            coeffs[i] = perm[k] => v
        end
    end
    # Next we renumber ch.prescribed_dofs, empty the dofmapping dict and then (re)close! it.
    # In close! the dependent fields (ch.free_dofs, ch.inhomogeneities,
    # ch.affine_inhomogeneities, ch.dofcoefficients) will automatically be permuted since
    # they are sorted based on ch.prescribed_dofs. The dofmapping is also updated in close!,
    # but it is necessary to empty it here since otherwise it might contain keys from the
    # old numbering.
    pdofs = ch.prescribed_dofs
    for i in eachindex(pdofs)
        pdofs[i] = perm[pdofs[i]]
    end
    empty!(ch.dofmapping)
    ch.closed[] = false
    close!(ch)
    return ch
end

######################
# Built-in orderings #
######################

# Renumber by a given permutation vector
function compute_renumber_permutation(dh::AbstractDofHandler, _, perm::AbstractVector{<:Integer})
    if !(isperm(perm) && length(perm) == ndofs(dh))
        error("input vector is not a permutation of length ndofs(dh)")
    end
    return perm
end

"""
    DofOrder.FieldWise()
    DofOrder.FieldWise(target_blocks::Vector{Int})

Dof order passed to [`renumber!`](@ref) to renumber global dofs field wise resulting in a
globally blocked system.

The default behavior is to group dofs of each field into their own block, with the same
order as in the DofHandler. This can be customized by passing a vector of length `nfields`
that maps each field to a "target block": to renumber a DofHandler with three fields `:u`,
`:v`, `:w` such that dofs for `:u` and `:w` end up in the first global block, and dofs for
`:v` in the second global block use `DofOrder.FieldWise([1, 2, 1])`.

This renumbering is stable such that the original relative ordering of dofs within each
target block is maintained.
"""
DofOrder.FieldWise

function compute_renumber_permutation(dh::Union{DofHandler, MixedDofHandler}, _, order::DofOrder.FieldWise)
    field_names = getfieldnames(dh)
    field_dims = map(fieldname -> getfielddim(dh, fieldname), dh.field_names)
    target_blocks = if isempty(order.target_blocks)
        Int[i for (i, dim) in pairs(field_dims) for _ in 1:dim]
    else
        if length(order.target_blocks) != length(field_names)
            error("length of target block vector does not match number of fields in DofHandler")
        end
        Int[order.target_blocks[i] for (i, dim) in pairs(field_dims) for _ in 1:dim]
    end
    return compute_renumber_permutation(dh, nothing, DofOrder.ComponentWise(target_blocks))
end

"""
    DofOrder.ComponentWise()
    DofOrder.ComponentWise(target_blocks::Vector{Int})

Dof order passed to [`renumber!`](@ref) to renumber global dofs component wise resulting in
a globally blocked system.

The default behavior is to group dofs of each component into their own block, with the same
order as in the DofHandler. This can be customized by passing a vector of length
`ncomponents` that maps each component to a "target block" (see [`DofOrder.FieldWise`](@ref)
for details).

This renumbering is stable such that the original relative ordering of dofs within each
target block is maintained.
"""
DofOrder.ComponentWise

function compute_renumber_permutation(dh::Union{DofHandler,MixedDofHandler}, _, order::DofOrder.ComponentWise)
    # Note: This assumes fields have the same dimension regardless of subdomain
    field_dims = map(fieldname -> getfielddim(dh, fieldname), dh.field_names)
    target_blocks = if isempty(order.target_blocks)
        collect(Int, 1:sum(field_dims))
    else
        if length(order.target_blocks) != sum(field_dims)
            error("length of target block vector does not match number of components in DofHandler")
        end
        order.target_blocks
    end
    @assert length(target_blocks) == sum(field_dims)
    @assert sort!(unique(target_blocks)) == 1:maximum(target_blocks)
    # Collect all dofs into the corresponding block according to target_blocks
    nblocks = maximum(target_blocks)
    dofs_for_blocks = [Set{Int}() for _ in 1:nblocks]
    component_offsets = pushfirst!(cumsum(field_dims), 0)
    flags = UpdateFlags(nodes=false, coords=false, dofs=true)
    for fh in (dh isa DofHandler ? (dh,) : dh.fieldhandlers)
        field_names = fh isa DofHandler ? fh.field_names : [f.name for f in fh.fields]
        dof_ranges = [dof_range(fh, f) for f in field_names]
        global_idxs = [findfirst(x -> x === f, dh.field_names) for f in field_names]
        set = dh isa DofHandler ? nothing : fh.cellset
        for cell in CellIterator(dh, set, flags)
            cdofs = celldofs(cell)
            for (local_idx, global_idx) in pairs(global_idxs)
                rng = dof_ranges[local_idx]
                fdim = field_dims[global_idx]
                component_offset = component_offsets[global_idx]
                for (j, J) in pairs(rng)
                    comp = mod1(j, fdim) + component_offset
                    block = target_blocks[comp]
                    push!(dofs_for_blocks[block], cdofs[J])
                end
            end
        end
    end
    @assert sum(length, dofs_for_blocks) == ndofs(dh)
    # Construct the inverse permutation. Sorting the dofs for each field is necessary to
    # make the permutation stable and keep internal ordering within each field, i.e. for
    # dofs i and j, if i > j, then p(i) > p(j), and if i < j, then p(i) < p(j) where p() is
    # the transformed dof number.
    iperm = sort!(collect(popfirst!(dofs_for_blocks)))
    sizehint!(iperm, ndofs(dh))
    for dofs_for_block in dofs_for_blocks
        append!(iperm, sort!(collect(dofs_for_block)))
    end
    # Construct permutation
    perm = invperm(iperm)
    return perm
end

function compute_renumber_permutation(dh::AbstractDofHandler, ::Union{ConstraintHandler,Nothing}, ::DofOrder.Ext{M}) where M
    error("Renumbering extension based on package $M not available.")
end
