Base.@deprecate_binding DirichletBoundaryConditions ConstraintHandler
Base.@deprecate_binding DirichletBoundaryCondition Dirichlet

import Base: push!
@deprecate push!(dh::AbstractDofHandler, args...) add!(dh, args...)

@deprecate vertices(ip::Interpolation) vertexdof_indices(ip) false
@deprecate faces(ip::Interpolation) facedof_indices(ip) false
@deprecate edges(ip::Interpolation) edgedof_indices(ip) false
@deprecate nfields(dh::AbstractDofHandler) length(getfieldnames(dh)) false

@deprecate add!(ch::ConstraintHandler, fh::FieldHandler, dbc::Dirichlet) add!(ch, dbc)


struct Cell{refdim, nnodes, nfaces}
    function Cell{refdim, nnodes, nfaces}(nodes) where {refdim, nnodes, nfaces}
        params = (refdim, nnodes, nfaces)
        replacement = nothing
        if params == (1, 2, 2) || params == (2, 2, 1) || params == (3, 2, 0)
            replacement = Line
        elseif params == (1, 3, 2)
            replacement = QuadraticLine
        elseif params == (2, 3, 3)
            replacement = Triangle
        elseif params == (2, 6, 3)
            replacement = QuadraticTriangle
        elseif params == (2, 4, 4) || params == (3, 4, 1)
            replacement = Quadrilateral
        elseif params == (2, 9, 4)
            replacement = QuadraticQuadrilateral
        elseif params == (2, 8, 4)
            replacement = SerendipityQuadraticQuadrilateral
        elseif params == (3, 4, 4)
            replacement = Tetrahedron
        elseif params == (3, 10, 4)
            replacement = QuadraticTetrahedron
        elseif params == (3, 8, 6)
            replacement = Hexahedron
        elseif params == (3, 27, 6)
            replacement = QuadraticHexahedron
        elseif params == (3, 20, 6)
            replacement = SerendipityQuadraticHexahedron
        elseif params == (3, 6, 5)
            replacement = Wedge
        end
        if replacement === nothing
            error("The AbstractCell interface have been changed, see https://github.com/Ferrite-FEM/Ferrite.jl/pull/679")
        else
            Base.depwarn("Use `$(replacement)(nodes)` instead of `Cell{$refdim, $nnodes, $nfaces}(nodes)`.", :Cell)
            return replacement(nodes)
        end
    end
end
export Cell

Base.@deprecate_binding Line2D Line
Base.@deprecate_binding Line3D Line
Base.@deprecate_binding Quadrilateral3D Quadrilateral
export Line2D, Line3D, Quadrilateral3D


import WriteVTK: vtk_grid, vtk_cell_data, vtk_point_data, vtk_save

@deprecate vtk_grid(filename::String, grid::AbstractGrid; kwargs...) open_vtk(filename, grid; kwargs...)
@deprecate vtk_grid(filename::String, dh::DofHandler; kwargs...) open_vtk(filename, dh; kwargs...)
@deprecate vtk_cell_data(vtks::VTKStream, args...) write_celldata(vtks, args...)
@deprecate vtk_point_data(vtks::VTKStream, dh::DofHandler, u, suffix="") (_vtk_write_solution(vtks.vtk, dh, u, suffix); vtks)
@deprecate vtk_point_data(vtks::VTKStream, data::Vector, args...) write_nodedata(vtks, data, args...)
@deprecate vtk_point_data(vtks::VTKStream, proj::L2Projector, args...) write_projected(vtks, proj, args...)
@deprecate vtk_point_data(vtks::VTKStream, ch::ConstraintHandler) write_dirichlet(vtks, ch)
@deprecate vtk_cellset(vtks::VTKStream, grid::AbstractGrid, args...) write_cellset(vtks, args...)
@deprecate vtk_nodeset(vtks::VTKStream, grid::AbstractGrid, args...) write_nodeset(vtks, args...)
@deprecate vtk_save(vtks::VTKStream) close(vtks)

@deprecate component_names(T) get_component_names(T) false # Internal function