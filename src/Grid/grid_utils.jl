

"""
    compute_bounding_box(grid::AbstractGrid)

Computes the bounding box for a given grid, based on its node coordinates. 
Returns the minimum and maximum vertices of the bounding box.
"""
function compute_bounding_box(grid::AbstractGrid{dim}) where {dim}
    T = get_coordinate_eltype(grid)
    min_vertex = Vec{dim}(i->T(+Inf))
    max_vertex = Vec{dim}(i->T(-Inf))
    for node in getnodes(grid)
        x = get_node_coordinate(node)
        max_vertex = Vec{dim}(i -> max(x[i], max_vertex[i]))
        min_vertex = Vec{dim}(i -> min(x[i], min_vertex[i]))
    end
    return min_vertex, max_vertex
end
