function get_cell_type(nen, ndim)
    # TODO: This is a bad way of figuring out the eltype
    if nen == 3 && ndim == 2
        cell =  VTKCellType.VTK_TRIANGLE
    elseif nen == 4 && ndim == 2
        cell = VTKCellType.VTK_QUAD
    elseif nen == 4 && ndim == 2
        cell = VTKCellType.VTK_HEXAHEDRON
    elseif nen == 4 && ndim == 3
        cell = VTKCellType.VTK_TETRA
    end
    return cell
end

function pad_zeros(points, ndim, nnodes)
    if ndim == 3
        points = points
    elseif ndim == 2
        points = [points; zeros(nnodes)']
    elseif ndim == 1
        points = [points; zeros(nnodes)'; zeros(nnodes)']
    end
    return points
end

"""
    vtk_grid(topology::Matrix{Int}, Coord::Matrix, filename::AbstractString) -> vtkgrid

Creates an unstructured VTK grid fromt the element topology and coordinates.

To add cell data and point data and write the file see https://github.com/jipolanco/WriteVTK.jl#generating-an-unstructured-vtk-file
"""
function vtk_grid(topology::Matrix{Int}, Coord::Matrix, filename::AbstractString)

    nele = size(topology, 2)
    nen = size(topology,1)
    nnodes = size(Coord, 2)
    ndim = size(Coord, 1)

    cell = get_cell_type(nen, ndim)

    points = Coord
    points = pad_zeros(points, ndim, nnodes)

    cells = MeshCell[MeshCell(cell, topology[:,i]) for i = 1:nele]

    vtk = vtk_grid(filename, points, cells)
    return vtk
end
