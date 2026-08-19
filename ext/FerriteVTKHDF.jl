# Implementation of `Ferrite.VTKHDFGridFile` (see src/Export/VTKHDF.jl), the
# VTKHDF counterpart of `VTKGridFile`, backed by VTKHDF.jl.
module FerriteVTKHDF

using Ferrite:
    Ferrite, VTKHDFGridFile, DofHandler, AbstractDofHandler, AbstractGrid,
    L2Projector, Vec, Tensor, SymmetricTensor,
    get_grid, getfieldnames, getnnodes, conformity, H1Conformity
import Ferrite: write_solution, write_projection, write_cell_data, write_node_data
using VTKHDF: VTKHDF, vtkhdf_grid, VTKPointData, VTKCellData

function Ferrite.VTKHDFGridFile(filename::String, dh::DofHandler; kwargs...)
    return VTKHDFGridFile(filename, get_grid(dh); write_discontinuous = Ferrite._vtk_discontinuous(dh), kwargs...)
end

function Ferrite.VTKHDFGridFile(filename::String, grid::AbstractGrid; write_discontinuous = false, kwargs...)
    if write_discontinuous
        coords, cls, cellnodes, node_mapping = Ferrite.create_discontinuous_vtk_griddata(grid)
    else
        coords, cls = Ferrite.create_vtk_griddata(grid)
        cellnodes = node_mapping = nothing
    end
    return VTKHDFGridFile(vtkhdf_grid(filename, coords, cls; kwargs...), cellnodes, node_mapping)
end

# do-block form
function Ferrite.VTKHDFGridFile(f::Function, args...; kwargs...)
    vtk = VTKHDFGridFile(args...; kwargs...)
    try
        f(vtk)
    finally
        close(vtk)
    end
    return vtk
end

hdf(vtk::VTKHDFGridFile) = vtk.vtk

function Base.close(vtk::VTKHDFGridFile)
    close(hdf(vtk))
    return vtk
end

function Base.show(io::IO, ::MIME"text/plain", vtk::VTKHDFGridFile)
    print(io, "VTKHDFGridFile (", hdf(vtk), ")")
    return
end

"""
    write_timestep(f::Function, vtk::VTKHDFGridFile, t::Real)

Write one time step to a `VTKHDFGridFile` opened with `temporal = true`. The
grid is stored once and shared by all steps. Inside the do-block the same
data functions as for static files are used:

```julia
write_timestep(vtkhdf, t) do vtk
    write_solution(vtk, dh, u)
end
```
"""
function VTKHDF.write_timestep(f::Function, vtk::VTKHDFGridFile, t::Real; kwargs...)
    VTKHDF.write_timestep(_ -> f(vtk), hdf(vtk), t; kwargs...)
    # flush each step so a hard kill (OOM, scheduler kill, segfault) still
    # leaves a valid file with all completed steps
    flush(hdf(vtk))
    return vtk
end

# Node data conversion mirrors the WriteVTK path, minus component names,
# which the VTKHDF format does not support.

function _node_matrix(nodedata::Vector{S}) where {O, D, T, M, S <: Union{Tensor{O, D, T, M}, SymmetricTensor{O, D, T, M}}}
    noutputs = S <: Vec{2} ? 3 : M # pad 2D Vec to 3D
    out = zeros(T, noutputs, length(nodedata))
    for i in eachindex(nodedata)
        Ferrite.toparaview!(@view(out[:, i]), nodedata[i])
    end
    return out
end

_write_node_array(vtk::VTKHDFGridFile, data::Vector{<:Ferrite.Tensors.AbstractTensor}, name::AbstractString) =
    setindex!(hdf(vtk), _node_matrix(data), name, VTKPointData())
_write_node_array(vtk::VTKHDFGridFile, data::Union{Vector{<:Real}, Matrix{<:Real}}, name::AbstractString) =
    setindex!(hdf(vtk), data, name, VTKPointData())

function write_solution(vtk::VTKHDFGridFile, dh::AbstractDofHandler, u::AbstractVector, suffix = "")
    fieldnames = getfieldnames(dh)
    for name in fieldnames
        if Ferrite.write_discontinuous(vtk)
            data = Ferrite.evaluate_at_discontinuous_vtkgrid_nodes(dh, u, name, vtk.cellnodes)
        else
            data = Ferrite._evaluate_at_grid_nodes(dh, u, name, #= vtk =# Val(true))
        end
        _write_node_array(vtk, data, string(name, suffix))
    end
    return vtk
end

function write_projection(vtk::VTKHDFGridFile, proj::L2Projector, vals, name)
    if Ferrite.write_discontinuous(vtk)
        data = Ferrite.evaluate_at_discontinuous_vtkgrid_nodes(proj.dh, vals, only(getfieldnames(proj.dh)), vtk.cellnodes)
    else
        data = Ferrite._evaluate_at_grid_nodes(proj, vals, #= vtk =# Val(true))::Matrix
        @assert size(data, 2) == getnnodes(get_grid(proj.dh))
    end
    _write_node_array(vtk, data, name)
    return vtk
end

function write_cell_data(vtk::VTKHDFGridFile, celldata, name)
    setindex!(hdf(vtk), celldata, name, VTKCellData())
    return vtk
end

function write_node_data(vtk::VTKHDFGridFile, nodedata, name)
    if Ferrite.write_discontinuous(vtk)
        data = Ferrite._map_to_discontinuous_nodes(vtk.node_mapping, nodedata)
    else
        data = nodedata
    end
    _write_node_array(vtk, data, name)
    return vtk
end

end # module FerriteVTKHDF
