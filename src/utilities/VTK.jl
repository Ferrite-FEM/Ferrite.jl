"""
    vtk_grid(Edof, Coord, Dof, nen, filename::AbstractString) -> vtkgrid
Creates an unstructured VTK grid. `nen` is the number of nodes per element

To add cell data and point data and write the file see
https://github.com/jipolanco/WriteVTK.jl#generating-an-unstructured-vtk-file
"""
function vtk_grid(Edof, Coord, Dof, nen, filename::AbstractString)
    top = topologyxtr(Edof,Coord,Dof,nen)

    nele = size(Edof, 2)
    nnodes = size(Coord, 2)
    ndim = size(Coord, 1)

    # TODO: This is a bad way of figuring out the eltype
    if nen == 3 && ndim == 2
        cell =  VTKCellType.VTK_TRIANGLE
    elseif nen == 4 && ndim == 2
        cell = VTKCellType.VTK_QUAD
    elseif nen == 4 && ndim == 2
        cell = VTKCellType.VTK_HEXAHEDRON
    end

    points = Coord

    if ndim == 2
        points = [points; zeros(nnodes)']
    elseif ndim == 1
        points = [points; zeros(nnodes)'; zeros(nnodes)']
    end

    cells = MeshCell[MeshCell(cell, top[:,i]) for i = 1:nele]

    vtk = vtk_grid(filename, points, cells)
    return vtk
end


