# PR #70
function vtk_grid{T}(topology::Matrix{Int}, coord::Matrix{T}, filename::AbstractString)
    Base.depwarn("vtk_grid(topology::Matrix{Int}, coord::Matrix, filename::AbstractString) is deprecated, use vtk_grid{dim,T}(filename::AbstractString, coords::Vector{Vec{dim,T}}, topology::Matrix{Int}, celltype::VTKCellTypes.VTKCellType) instead", :vtk_grid)
    nen = size(topology,1)
    nnodes = size(coord, 2)
    ndim = size(coord, 1)

    if nen == 3 && ndim == 2
        cell =  VTKCellTypes.VTK_TRIANGLE
    elseif nen == 4 && ndim == 2
        cell = VTKCellTypes.VTK_QUAD
    elseif nen == 8 && ndim == 3
        cell = VTKCellTypes.VTK_HEXAHEDRON
    elseif nen == 4 && ndim == 3
        cell = VTKCellTypes.VTK_TETRA
    end

    coords = reinterpret(Vec{ndim,T},coord,(nnodes,))

    vtk = vtk_grid(filename, coords, topology, cell)
    return vtk
end

# Issue #74, PR #76
immutable Dim{T} end
export Dim
