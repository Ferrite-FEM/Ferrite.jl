
using Ferrite

struct DofTools{DH}
    dh::DH
    local_facedofs::Ferrite.CellVector{Int}
    local_edgedofs::Ferrite.CellVector{Int}
    local_vertexdofs::Ferrite.CellVector{Int}

    local_edge_coords::Ferrite.CellVector{Int}
end

function DofTools(dh::Ferrite.AbstractDofHandler)

    #@assert( length(dh.fieldhandlers) == 1)

    lf = _extract_local_dofs(dh, Ferrite.boundaryfunction(FaceIndex))
    le = _extract_local_dofs(dh, Ferrite.boundaryfunction(EdgeIndex))
    lv = _extract_local_dofs(dh, Ferrite.boundaryfunction(VertexIndex))

    lec = _extract_local_coords(dh, Ferrite.boundaryfunction(EdgeIndex))

    return DofTools(dh, lf, le, lv, lec)
end

function _extract_local_dofs(dh::Ferrite.AbstractDofHandler, faces::Function)

    local_facedofs = Ferrite.CellVector(Int[],Int[],Int[])
    nfaces = length( faces( Ferrite.default_interpolation( getcelltype(dh.grid) ) ))

    nfields = length(Ferrite.getfieldnames(dh))

    for iface in 1:nfaces
        local_face_dofs = []
        offset = 0
        for i in 1:nfields
            ip = Ferrite.getfieldinterpolation(dh, i)
            field_dim = Ferrite.getfielddim(dh, i)
            field_faces = faces(ip)
            @assert(length(field_faces) == nfaces)
            for fdof in field_faces[iface], d in 1:field_dim
                push!(local_face_dofs, (fdof-1)*field_dim + d + offset)
            end
            offset += getnbasefunctions(ip)*field_dim
        end
        push!(local_facedofs.offset, length(local_facedofs.values)+1)
        push!(local_facedofs.length, length(local_face_dofs))
        append!(local_facedofs.values, local_face_dofs)
    end

    return local_facedofs

end

function _extract_local_coords(dh::Ferrite.AbstractDofHandler, faces::Function)

    local_facedofs = Ferrite.CellVector(Int[],Int[],Int[])
    ip_geom = Ferrite.default_interpolation( getcelltype(dh.grid) )
    field_faces = faces(ip_geom)
    nfaces = length( field_faces )

    for iface in 1:nfaces
        local_face_dofs = []
        for fdof in field_faces[iface]
            push!(local_face_dofs, fdof)
        end
        push!(local_facedofs.offset, length(local_facedofs.values)+1)
        push!(local_facedofs.length, length(local_face_dofs))
        append!(local_facedofs.values, local_face_dofs)
    end

    return local_facedofs

end

getnfacebasefunctions(ip, faceid::Int=1) = length(Ferrite.faces(ip)[faceid])::Int

field_offset_face(dh::DofHandler, field_name::Symbol, faceid::Int=1) = field_offset_face(dh, Ferrite.find_field(dh, field_name), faceid)
function field_offset_face(dh::DofHandler, fieldidx::Int, faceid::Int=1)

    offset = 0
    for i in 1:fieldidx-1
        ip = dh.field_interpolations[i]
        offset += getnfacebasefunctions(ip, faceid) * dh.field_dims[i]
    end
    return offset
end

function dof_range_face(dh::DofHandler, field_name::Symbol)

    fieldidx = Ferrite.find_field(dh, field_name)
    offset = field_offset_face(dh, fieldidx)
    ip = dh.field_interpolations[fieldidx]
    fdim = dh.field_dims[fieldidx]
    n_field_dofs = getnfacebasefunctions(ip)::Int * fdim

    return (offset+1):(offset+n_field_dofs)

end

function facedofs!(_facedofs::Vector{Int}, dt::DofTools, faceid::FaceIndex, _celldofs::Vector{Int})
    dh = dt.dh
    cellid, faceidx = faceid
    celldofs!(_celldofs, dh, cellid)
    for (i,d) in enumerate(dt.local_edgedofs[faceidx])
        _facedofs[i] = _celldofs[d] 
    end
    return _facedofs
end

function edgedofs!(global_face_dofs::Vector{Int}, dt::DofTools, edgeid::EdgeIndex)
    dh = dt.dh
    cellid, edgeidx = edgeid
    o = dh.cell_dofs.offset[cellid]-1
    for (i,d) in enumerate(dt.local_edgedofs[edgeidx])
        global_face_dofs[i] = dh.cell_dofs.values[o+d] 
    end
    return global_face_dofs
end

function vertexdofs!(global_face_dofs::Vector{Int}, dt::DofTools, edgeid::VertexIndex)
    dh = dt.dh
    cellid, edgeidx = edgeid
    o = dh.cell_dofs.offset[cellid]-1
    for (i,d) in enumerate(dt.local_vertexdofs[edgeidx])
        global_face_dofs[i] = dh.cell_dofs.values[o+d] 
    end
    return global_face_dofs
end

function edgecoords!(global_coords::Vector{Vec{dim,T}}, dt::DofTools, edgeid::EdgeIndex) where {dim,T}
    dh = dt.dh
    cellid, edgeidx = edgeid
    o = dh.cell_coords.offset[cellid]-1
    for (i,d) in enumerate(dt.local_edge_coords[edgeidx])
        global_coords[i] = dh.cell_coords.values[o+d] 
    end
    return global_coords
end

function get_dofs_on_boundary(dh::DofHandler, set::Set{FaceIndex}, field::Symbol, components::AbstractVector{Int})
    field_range = dof_range_face(dh, field)
    dofs_on_face = Int[]
	for faceidx in set
		facedofs!(fdofs, dh, faceidx)
        for r in field_range, c in components
            push!(dofs_on_face, fdofs[r+c])
        end
	end

    unique!(dofs_on_face)
end



function test()

    grid = generate_grid(Quadrilateral, (1,2))
    dh = DofHandler(grid)
    push!(dh, :u, 2,  Lagrange{2,RefCube,2}()) # displacement
    push!(dh, :p, 1,  Lagrange{2,RefCube,1}()) # pressure
    close!(dh)

    dt = DofTools(dh)
end