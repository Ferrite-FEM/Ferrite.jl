# Utilities for testing the VTK export by reading back the written files with ReadVTK.jl

using ReadVTK: ReadVTK, get_cells, get_data, get_points

# Read the `.vtu` file `filename` and return a named tuple with the fields
#  - `points`: the coordinates of the points, as a `3 × npoints` matrix
#  - `cells`: the (one-based) point numbers of each cell
#  - `celltypes`: the VTK cell type id of each cell
#  - `point_data`, `cell_data`: dictionaries mapping the name of each data array to its values,
#    as a vector for one component and as a `ncomponents × n` matrix otherwise
#  - `component_names`: dictionary mapping the name of each data array to its component names
#    (`nothing` if not given)
function read_vtk(filename)
    vtk = ReadVTK.VTKFile(filename)
    vtkcells = get_cells(vtk)
    starts = [1; vtkcells.offsets[1:(end - 1)] .+ 1]
    cells = [vtkcells.connectivity[i:j] for (i, j) in zip(starts, vtkcells.offsets)]
    point_data, point_names = _read_vtk_data(ReadVTK.get_point_data, vtk)
    cell_data, cell_names = _read_vtk_data(ReadVTK.get_cell_data, vtk)
    return (;
        points = collect(get_points(vtk)), cells, celltypes = collect(vtkcells.types),
        point_data, cell_data, component_names = merge(point_names, cell_names),
    )
end

function _read_vtk_data(get_vtk_data, vtk)
    data = Dict{String, Array}()
    names = Dict{String, Union{Nothing, Vector{String}}}()
    # ReadVTK throws a BoundsError if the file has no data of this kind
    vtk_data = try
        get_vtk_data(vtk)
    catch err
        err isa BoundsError || rethrow()
        return data, names
    end
    for name in keys(vtk_data)
        @assert !haskey(data, name) "duplicate data array $(name)"
        array = vtk_data[name]
        data[name] = collect(get_data(array))
        xml = array.data_array
        ncomponents = parse(Int, something(ReadVTK.LightXML.attribute(xml, "NumberOfComponents"), "1"))
        component_names = [ReadVTK.LightXML.attribute(xml, "ComponentName$(i - 1)") for i in 1:ncomponents]
        names[name] = all(isnothing, component_names) ? nothing : String.(component_names)
    end
    return data, names
end

# The permutation from the Ferrite node order to the VTK node order, for the cells where they differ
const VTK_NODE_ORDER = Dict(
    Pyramid => [1, 2, 4, 3, 5],
    QuadraticWedge => [1, 2, 3, 4, 5, 6, 7, 10, 8, 13, 15, 14, 9, 11, 12, 16, 18, 17],
    QuadraticHexahedron => [1:20; 25; 23; 22; 24; 21; 26; 27],
)
vtk_node_order(cell::Ferrite.AbstractCell) = get(VTK_NODE_ORDER, typeof(cell), 1:Ferrite.nnodes(cell))

# Values as written to a VTK file: scalars as a vector, and vectors and tensors as a
# `ncomponents × n` matrix with 2D vectors padded with zeros and tensors in Voigt order
vtk_values(x::AbstractVector{<:Number}) = collect(x)
vtk_values(x::AbstractVector{<:Vec{1}}) = [v[1] for v in x]
vtk_values(x::AbstractVector{<:Vec{2}}) = reduce(hcat, [v[1], v[2], zero(eltype(v))] for v in x)
vtk_values(x::AbstractVector{<:Vec{3}}) = reduce(hcat, collect(v) for v in x)
vtk_values(x::AbstractVector{<:Ferrite.SecondOrderTensor}) = reduce(hcat, tovoigt(t) for t in x)

# Test that the points, cells and cell types read with `read_vtk` match `grid`. If
# `discontinuous`, the points are expected to be duplicated for each cell. Return the grid node
# number of each point.
function test_vtk_grid(data, grid; discontinuous = false)
    cells = getcells(grid)
    if discontinuous
        point_nodes = reduce(vcat, collect(Ferrite.get_node_ids(cell)) for cell in cells)
        offsets = cumsum([0; [Ferrite.nnodes(cell) for cell in cells]])
        expected_cells = [offsets[i] .+ vtk_node_order(cell) for (i, cell) in enumerate(cells)]
    else
        point_nodes = collect(1:getnnodes(grid))
        expected_cells = [collect(Ferrite.get_node_ids(cell))[vtk_node_order(cell)] for cell in cells]
    end
    # The points are always written in 3D
    vtk_point(x) = [i <= length(x) ? x[i] : zero(eltype(x)) for i in 1:3]
    @test data.points == reduce(hcat, vtk_point(get_node_coordinate(grid, n)) for n in point_nodes)
    @test data.cells == expected_cells
    @test data.celltypes == [Ferrite.cell_to_vtkcell(typeof(cell)).vtk_id for cell in cells]
    return point_nodes
end
