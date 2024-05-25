using Ferrite, OrderedCollections

function create_new_cellset(old_set::OrderedSet{Int}, cellmapping::Vector)
    set = sizehint!(OrderedSet{Int}(), length(old_set))
    for i in old_set
        for j in cellmapping[i]
            j != 0 && push!(set, j)
        end
    end
    return set
end

function create_new_facetset(old_set::OrderedSet{FacetIndex}, cellmapping::Vector{NTuple{N, Int}}, OldCellType, NewCellType) where N
    set = sizehint!(OrderedSet{FacetIndex}(), length(old_set))
    for (cellnr, facetnr) in old_set
        if last(cellmapping[cellnr]) == 0 # Cell has not changed, add same facet index
            push!(set, FacetIndex(cellmapping[cellnr][1], facetnr))
        else
            push!(set, create_new_facet(OldCellType, NewCellType, cellmapping[cellnr], facetnr))
        end
    end
    return set
end

function create_new_cells(cell::Quadrilateral, ::Type{Triangle})
    # 4---3      4---3
    # |   |  =>  |2/1|
    # 1---2      1---2
    ns = Ferrite.get_node_ids(cell)
    return (Triangle((ns[1], ns[2], ns[3])), Triangle((ns[1], ns[3], ns[4])))
end

function create_new_facet(::Type{Quadrilateral}, ::Type{Triangle}, cellnrs::NTuple{2, Int}, old_facetnr::Int)
    # Defined following the split done in `create_new_cells`
    old_facetnr == 1 && return FacetIndex(cellnrs[1], 1)
    old_facetnr == 2 && return FacetIndex(cellnrs[1], 2)
    old_facetnr == 3 && return FacetIndex(cellnrs[2], 2)
    old_facetnr == 4 && return FacetIndex(cellnrs[2], 3)
    throw(ArgumentError("facetnr ∉ (1,2,3,4)"))
end

function create_mixed_grid(grid::Grid{<:Any, Quadrilateral}, ::Type{Triangle}, set::AbstractSet{Int})
    # One old Quadrilateral maps to 2 new Triangle cells
    cellmapping = Vector{NTuple{2, Int}}(undef, getncells(grid)) # old_id => new_ids
    cells = Union{Quadrilateral, Triangle}[]
    nr = 1
    for (i, cell) in enumerate(getcells(grid))
        if i ∈ set
            new_cells = create_new_cells(cell, Triangle)
            cellmapping[i] = (nr, nr + 1)
            for new_cell in new_cells
                push!(cells, new_cell)
            end
            nr += length(new_cells)
        else
            push!(cells, cell)
            cellmapping[i] = (nr, 0)
            nr += 1
        end
    end

    isempty(grid.vertexsets) || @warn "vertexsets are ignored"

    cellsets = Dict(key => create_new_cellset(set, cellmapping) for (key, set) in grid.cellsets)
    facetsets = Dict(key => create_new_facetset(set, cellmapping, Quadrilateral, Triangle) for (key, set) in grid.facetsets)
    return Grid(cells, grid.nodes; cellsets, facetsets, nodesets = grid.nodesets)
end

function testit()
    grid = generate_grid(Quadrilateral, (10, 10))

    addcellset!(grid, "all_old_cells", 1:getncells(grid))
    addcellset!(grid, "changed_cells", x -> x[1] > 0 && x[2] < 0)

    new_grid = create_mixed_grid(grid, Triangle, getcellset(grid, "changed_cells"))

    # Export
    VTKFile("oldgrid", grid) do vtk
        Ferrite.write_cellset(vtk, grid, keys(grid.cellsets))
    end

    dh = DofHandler(new_grid)
    sdh1 = SubDofHandler(dh, getcellset(new_grid, "changed_cells"))
    add!(sdh1, :u, Lagrange{RefTriangle,1}())
    sdh2 = SubDofHandler(dh, setdiff(1:getncells(new_grid), getcellset(new_grid, "changed_cells")))
    add!(sdh2, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(new_grid, "right"), Returns(0.0)))
    close!(ch)

    VTKFile("newgrid", new_grid) do vtk
        Ferrite.write_cellset(vtk, new_grid, keys(new_grid.cellsets))
        Ferrite.write_constraints(vtk, ch)
    end
    return nothing
end
