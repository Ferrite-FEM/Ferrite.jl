"""
    bounding_box(grid::AbstractGrid)

Computes the axis-aligned bounding box for a given grid, based on its node coordinates. 
Returns the minimum and maximum vertex coordinates of the bounding box.
"""
function bounding_box(grid::AbstractGrid{dim}) where {dim}
    T = get_coordinate_eltype(grid)
    min_vertex = Vec{dim}(i->typemax(T))
    max_vertex = Vec{dim}(i->typemin(T))
    for node in getnodes(grid)
        x = get_node_coordinate(node)
        _max_tmp = max_vertex # avoid type instability
        _min_tmp = min_vertex
        max_vertex = Vec{dim}(i -> max(x[i], _max_tmp[i]))
        min_vertex = Vec{dim}(i -> min(x[i], _min_tmp[i]))
    end
    return min_vertex, max_vertex
end
