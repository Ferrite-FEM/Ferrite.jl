struct PeriodicDofPosMapping
    local_facet_dofpos::Vector{Int}
    local_facet_dofpos_offset::Vector{Int}
    rotated_indices::Matrix{Int}
    mirrored_indices::Vector{Int}
end

"""
    PeriodicDofPosMapping(func_interpolation::Interpolation, field_dof_offset)

This helper struct makes it easy to get the corresponding DoF positions between two facets, which may be
mirrored and/or rotated relative to eachother. A dof position corresponds to the local (cell) dof value for
standard interpolations, but for `VectorizedInterpolation`s, it corresponds to the local dof value for the
first component. The functions `get_dof_locations` and `get_local_dof_pair` can be used to get the mapping between
the *mirror* and *image* DoF, e.g.
```
grid = generate_grid(Triangle, (3, 3))
ip = Lagrange{RefTriangle, 2}()^2
dh = close!(add!(DofHandler(grid), :u, ip))
facet_pair = collect_periodic_facets(grid, "right", "left")[1]
field_dof_offset = 0 # Consider a single field problem (or first field if multifield)
dofmap = Ferrite.PeriodicDofPosMapping(ip, field_dof_offset)
for dof_location in Ferrite.get_dof_locations(dofmap, facet_pair)
    mdof, idof = Ferrite.get_local_dof_pair(dofmap, facet_pair, dof_location)
    mirror_dof = celldofs(dh, facet_pair.mirror[1])[mdof]
    image_dof = celldofs(dh, facet_pair.image[1])[idof]
    println("DoF ", mirror_dof, " should mirror DoF ", image_dof)
end
```
"""
function PeriodicDofPosMapping(interpolation, field_dof_offset)
    local_facet_dofpos, local_facet_dofpos_offset = pu_local_facet_dofpos_for_bc(interpolation, field_dof_offset, FacetIndex)
    ipf_base = get_base_interpolation(interpolation)
    mirrored_indices = pu_mirror_local_facetdofs(local_facet_dofpos_offset, ipf_base)
    rotated_indices = pu_rotate_local_facetdofs(local_facet_dofpos_offset, ipf_base)
    return PeriodicDofPosMapping(local_facet_dofpos, local_facet_dofpos_offset, rotated_indices, mirrored_indices)
end

function get_dof_locations(dofmap::PeriodicDofPosMapping, facet_pair::PeriodicFacetPair)
    mirror_facetnr = facet_pair.mirror[2]
    offsets = dofmap.local_facet_dofpos_offset
    return 1:(offsets[mirror_facetnr + 1] - offsets[mirror_facetnr])
end

function get_local_dof_pair(dofmap::PeriodicDofPosMapping, facet_pair::PeriodicFacetPair, dof_location::Int)
    m = facet_pair.mirror
    i = facet_pair.image

    md = dofmap.local_facet_dofpos_offset[m[2]] + dof_location - 1
    id = dofmap.local_facet_dofpos_offset[i[2]] + dof_location - 1

    # Rotate the mirror index
    rotated_md = dofmap.rotated_indices[md, facet_pair.rotation + 1]
    # Mirror the mirror index (maybe) :)
    mirrored_md = facet_pair.mirrored ? dofmap.mirrored_indices[rotated_md] : rotated_md
    idof = dofmap.local_facet_dofpos[id]
    mdof = dofmap.local_facet_dofpos[mirrored_md]
    return mdof => idof
end
# Convenience (better to use above version and manually add ` component - 1`)
function get_local_dof_pair(dofmap::PeriodicDofPosMapping, facet_pair::PeriodicFacetPair, dof_location::Int, component::Int)
    mdof, idof = get_local_dof_pair(dofmap, facet_pair, dof_location)
    Δn = component - 1
    return (mdof + Δn) => (idof + Δn)
end

function pu_local_facet_dofpos_for_bc(interpolation, field_dof_offset, ::Type{BI}) where {BI <: BoundaryIndex}
    ipf_base = get_base_interpolation(interpolation)
    n_dbc_comp = n_dbc_components(interpolation)

    local_facet_dofpos = Int[]
    local_facet_dofpos_offset = Int[1]
    for boundarydofs in dirichlet_boundarydof_indices(BI)(ipf_base)
        for boundarydof in boundarydofs
            push!(local_facet_dofpos, 1 + (boundarydof - 1) * n_dbc_comp + field_dof_offset)
        end
        push!(local_facet_dofpos_offset, length(local_facet_dofpos) + 1)
    end
    return local_facet_dofpos, local_facet_dofpos_offset
end

function pu_mirror_local_facetdofs(_, ::Lagrange{RefLine})
    return ones(Int, 1) # Nothing to mirror
end
function pu_mirror_local_facetdofs(local_facet_dofpos_offset, ip::Lagrange{<:Union{RefQuadrilateral, RefTriangle}})
    @assert getorder(ip) ≤ 2
    # For 2D we always permute since Ferrite defines dofs counter-clockwise
    ret = collect(1:(local_facet_dofpos_offset[end] - 1))
    for facetnr in 1:nfacets(ip)
        # Mirror the vertex order
        v1_offset = local_facet_dofpos_offset[facetnr]
        v2_offset = v1_offset + 1
        ret[v1_offset] = v2_offset
        ret[v2_offset] = v1_offset
        # TODO: Higher order has more than one edgedof, and this would require mirroring as well.
    end
    return ret
end

# TODO: Can probably be combined with the method above.
function pu_mirror_local_facetdofs(local_facet_dofpos_offset, ip::Lagrange{<:Union{RefHexahedron, RefTetrahedron}, order}) where {order}
    @assert 1 <= order <= 2
    N = ip isa Lagrange{RefHexahedron} ? 4 : 3
    ret = collect(1:(local_facet_dofpos_offset[end] - 1))

    # Mirror by changing from counter-clockwise to clockwise
    for facetnr in 1:nfacets(ip)
        r = local_facet_dofpos_offset[facetnr]:(local_facet_dofpos_offset[facetnr + 1] - 1)
        # 1. Rotate the corners
        vertex_range = r[1:N]
        vlr = @view ret[vertex_range]
        reverse!(vlr)
        circshift!(vlr, 1)
        # 2. Rotate the edge dofs for quadratic interpolation
        if order > 1
            edge_range = r[(N + 1):2N]
            elr = @view ret[edge_range]
            reverse!(elr)
            # circshift!(elr, 1) # !!! Note: no shift here
        end
    end
    return ret
end

function pu_rotate_local_facetdofs(local_facet_dofpos_offset, ip::Lagrange{<:Union{RefQuadrilateral, RefTriangle}})
    ret = zeros(Int, local_facet_dofpos_offset[end] - 1, 1)
    ret .= 1:(local_facet_dofpos_offset[end] - 1)
    return ret
end

function pu_rotate_local_facetdofs(local_facet_dofpos_offset, ip::Lagrange{<:Union{RefHexahedron, RefTetrahedron}, order}) where {order}
    @assert 1 <= order <= 2
    N = ip isa Lagrange{RefHexahedron} ? 4 : 3
    ret = zeros(Int, local_facet_dofpos_offset[end] - 1, N)
    ret[:, :] .= 1:(local_facet_dofpos_offset[end] - 1)
    for f in 1:(length(local_facet_dofpos_offset) - 1)
        facet_range = local_facet_dofpos_offset[f]:(local_facet_dofpos_offset[f + 1] - 1)
        for i in 1:(N - 1)
            # 1. Rotate the vertex dofs
            vertex_range = facet_range[1:N]
            circshift!(@view(ret[vertex_range, i + 1]), @view(ret[vertex_range, i]), -1)
            # 2. Rotate the edge dofs
            if order > 1
                edge_range = facet_range[(N + 1):2N]
                circshift!(@view(ret[edge_range, i + 1]), @view(ret[edge_range, i]), -1)
            end
        end
    end
    return ret
end


struct PeriodicDofLocations
    local_facet_dofpos_offset::Vector{Int}
    rotated_indices::Matrix{Int}
    mirrored_indices::Vector{Int}
end

_facetrange(dm::PeriodicDofLocations, facetnr) = dm.local_facet_dofpos_offset[facetnr]:(dm.local_facet_dofpos_offset[facetnr + 1] - 1)
get_rotation_indices(dm::PeriodicDofLocations, facetnr::Integer, rotation::Integer) = view(dm.rotated_indices, _facetrange(dm, facetnr), rotation + 1)
get_mirrored_indices(dm::PeriodicDofLocations, facetnr::Integer) = view(dm.mirrored_indices, _facetrange(dm, facetnr))

function PeriodicDofLocations(interpolation)
    ipf_base = get_base_interpolation(interpolation)
    local_facet_dofpos_offset = ones(Int, nfacets(ipf_base) + 1)
    for (facetnr, facetdofs) in pairs(dirichlet_facetdof_indices(ipf_base))
        local_facet_dofpos_offset[facetnr + 1] = local_facet_dofpos_offset[facetnr] + length(facetdofs)
    end
    max_num_facet_rotations = maximum(length, reference_facets(getrefshape(interpolation)))
    mirrored_indices = zeros(Int, local_facet_dofpos_offset[end] - 1)
    rotated_indices = zeros(Int, local_facet_dofpos_offset[end] - 1, max_num_facet_rotations)
    pdloc = PeriodicDofLocations(local_facet_dofpos_offset, rotated_indices, mirrored_indices)
    compute_mirror_indices!(pdloc, ipf_base)
    compute_rotated_indices!(pdloc, ipf_base)
    return pdloc
end

function compute_mirror_indices!(::PeriodicDofLocations, ::Lagrange{RefLine})
    return ones(Int, 1) # Nothing to mirror
end
function compute_mirror_indices!(pdl::PeriodicDofLocations, ip::Lagrange{<:Union{RefQuadrilateral, RefTriangle}})
    for facetnr in 1:nfacets(ip)
        inds = get_mirrored_indices(pdl, facetnr)
        inds[1:2] .= (2, 1) # Reverse vertex dofs
        inds[3:end] .= length(inds):-1:3
    end
    return pdl
end
function compute_mirror_indices!(pdl::PeriodicDofLocations, ip::Lagrange{<:Union{RefHexahedron, RefTetrahedron}, order}) where {order}
    @assert 1 <= order <= 2

    # Mirror by changing from counter-clockwise to clockwise
    for (facetnr, nverts) in pairs(length.(reference_facets(getrefshape(ip))))
        inds = get_mirrored_indices(pdl, facetnr)
        # 1. Rotate the vertices
        vertex_range = 1:nverts
        vlr = @view inds[vertex_range]
        reverse!(vlr)
        circshift!(vlr, 1)
        # 2. Rotate the edge dofs for quadratic interpolation
        if order === 2
            edge_range = (nverts + 1):(2 * nverts)
            elr = @view inds[edge_range]
            reverse!(elr)
            # circshift!(elr, 1) # !!! Note: no shift here
        end
    end
    return pdl
end

function compute_rotated_indices!(pdl::PeriodicDofLocations, ip::Lagrange{<:Union{RefQuadrilateral, RefTriangle}})
    # No rotation in 2D
    for facetnr in 1:nfacets(ip)
        inds = get_rotation_indices(pdl, facetnr, 0)
        inds .= 1:length(inds)
    end
    return pdl
end

function compute_rotated_indices!(pdl::PeriodicDofLocations, ip::Lagrange{<:Union{RefHexahedron, RefTetrahedron}, order}) where {order}
    @assert 1 <= order <= 2
    nvertices_per_facet = length.(reference_facets(getrefshape(ip)))
    for (facetnr, nverts) in pairs(nvertices_per_facet)
        inds = get_rotation_indices(pdl, facetnr, 0) # Zero rotation
        inds .= 1:length(inds)
        vertex_range = 1:nverts
        edge_range = (nverts + 1):(2 * nverts)
        for i in 1:(nverts - 1) # Equally many rotations possible as there are vertices on a facet
            # 1. Rotate the vertex dofs
            circshift!(
                view(get_rotation_indices(pdl, facetnr, i), vertex_range),
                view(get_rotation_indices(pdl, facetnr, i - 1), vertex_range), -1
            )
            # 2. Rotate the edge dofs
            if order == 2
                circshift!(
                    view(get_rotation_indices(pdl, facetnr, i), edge_range),
                    view(get_rotation_indices(pdl, facetnr, i - 1), edge_range), -1
                )
            end
        end
    end
    return pdl
end

function get_mirror_dof_location(pdl::PeriodicDofLocations, fp::PeriodicFacetPair, image_dof_location::Integer)
    facetnr = fp.mirror[2]
    rotated_dof_location = get_rotation_indices(pdl, facetnr, fp.rotation)[image_dof_location]
    mirrored_dof_location = fp.mirrored ? get_mirrored_indices(pdl, facetnr)[rotated_dof_location] : rotated_dof_location
    return mirrored_dof_location
end
