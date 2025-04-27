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
        inds .= 1:length(inds)
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
            if order == 2
                # 2. Rotate the edge dofs
                circshift!(
                    view(get_rotation_indices(pdl, facetnr, i), edge_range),
                    view(get_rotation_indices(pdl, facetnr, i - 1), edge_range), -1
                )
                # 3. No change for face dof for Lagrange{RefHexahedron,2}
                if ip isa Lagrange{RefHexahedron, 2}
                    get_rotation_indices(pdl, facetnr, i)[end] = get_rotation_indices(pdl, facetnr, i - 1)[end]
                end
            end
        end
    end
    return pdl
end

function get_mirror_dof_location(pdl::PeriodicDofLocations, fp::PeriodicFacetPair, image_dof_location::DofLocation)
    locnr = image_dof_location.location_nr
    facetnr = fp.mirror[2]
    rotated_dof_location = get_rotation_indices(pdl, facetnr, fp.rotation)[locnr]
    mirrored_dof_location = fp.mirrored ? get_mirrored_indices(pdl, facetnr)[rotated_dof_location] : rotated_dof_location
    return DofLocation(facetnr, mirrored_dof_location)
end
