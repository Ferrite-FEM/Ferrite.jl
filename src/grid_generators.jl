export generate_grid
"""
`Grid` generator for a rectangle in 1, 2 and 3 dimensions.

    generate_grid(celltype::Cell{dim, N}, nel::NTuple{dim, Int}, [left::Vec{1, T}=Vec{1}((-1.0,)), right::Vec{1, T}=Vec{1}((1.0,))])

**Arguments**

* `celltype`: a celltype, e.g. `Triangle` or `Hexahedron`
* `nel`: a tuple with number of elements in each direction.
* `left`, `right`: optional endpoints of the domain, defaults to `-one(Vec{dim})` and `one(Vec{dim})`

**Results**

* `grid`: a `Grid`.

"""
# Line
function generate_grid{T}(::Type{Line}, nel::NTuple{1, Int}, left::Vec{1, T}=Vec{1}((-1.0,)), right::Vec{1, T}=Vec{1}((1.0,)))
    nel_x = nel[1]
    n_nodes = nel_x + 1

    # Generate nodes
    coords_x = collect(linspace(left[1], right[1], n_nodes))
    nodes = Node{1,T}[]
    for i in 1:n_nodes
        push!(nodes, Node((coords_x[i],)))
    end

    # Generate cells
    cells = Line[]
    for i in 1:nel_x
        push!(cells, Line((i, i+1)))
    end

    # Cell faces
    boundary = CellFace[CellFace((1, 1)),
                              CellFace((nel_x, 2))]

    # Cell face sets
    facesets = Dict("left"  => [1],
                         "right" => [2])
    return Grid(cells, nodes, boundary=boundary, facesets=facesets)
end

# QuadraticLine
function generate_grid{T}(::Type{QuadraticLine}, nel::NTuple{1, Int}, left::Vec{1, T}=Vec{1}((-1.0,)), right::Vec{1, T}=Vec{1}((1.0,)))
    nel_x = nel[1]
    n_nodes = 2*nel_x + 1

    # Generate nodes
    coords_x = collect(linspace(left[1], right[1], n_nodes))
    nodes = Node{1,T}[]
    for i in 1:n_nodes
        push!(nodes, Node((coords_x[i],)))
    end

    # Generate cells
    cells = QuadraticLine[]
    for i in 1:nel_x
        push!(cells, QuadraticLine((2*i-1, 2*i+1, 2*i)))
    end

    # Cell faces
    boundary = CellFace[CellFace((1, 1)),
                              CellFace((nel_x, 2))]

    # Cell face sets
    facesets = Dict("left"  => [1],
                         "right" => [2])
    return Grid(cells, nodes, boundary=boundary, facesets=facesets)
end

# Quadrilateral
function generate_grid{T}(::Type{Quadrilateral}, nel::NTuple{2, Int}, left::Vec{2, T}=Vec{2}((-1.0,-1.0)), right::Vec{2, T}=Vec{2}((1.0,1.0)))
    nel_x = nel[1]; nel_y = nel[2]; nel_tot = nel_x*nel_y
    n_nodes_x = nel_x + 1; n_nodes_y = nel_y + 1
    n_nodes = n_nodes_x * n_nodes_y

    # Generate nodes
    coords_x = linspace(left[1], right[1], n_nodes_x)
    coords_y = linspace(left[2], right[2], n_nodes_y)
    nodes = Node{2,T}[]
    for j in 1:n_nodes_y, i in 1:n_nodes_x
        push!(nodes, Node((coords_x[i], coords_y[j])))
    end

    # Generate cells
    node_array = reshape(collect(1:n_nodes), (n_nodes_x, n_nodes_y))
    cells = Quadrilateral[]
    for j in 1:nel_y, i in 1:nel_x
        push!(cells, Quadrilateral((node_array[i,j], node_array[i+1,j], node_array[i+1,j+1], node_array[i,j+1])))
    end

    # Cell faces
    cell_array = reshape(collect(1:nel_tot),(nel_x, nel_y))
    boundary = CellFace[[CellFace((cl, 1)) for cl in cell_array[:,1]];
                              [CellFace((cl, 2)) for cl in cell_array[end,:]];
                              [CellFace((cl, 3)) for cl in cell_array[:,end]];
                              [CellFace((cl, 4)) for cl in cell_array[1,:]]]

    # Cell face sets
    offset = 0
    facesets = Dict{String, Vector{Int}}()
    facesets["bottom"] = (1:length(cell_array[:,1]))   + offset; offset += length(cell_array[:,1])
    facesets["right"]  = (1:length(cell_array[end,:])) + offset; offset += length(cell_array[end,:])
    facesets["top"]    = (1:length(cell_array[:,end])) + offset; offset += length(cell_array[:,end])
    facesets["left"]   = (1:length(cell_array[1,:]))   + offset; offset += length(cell_array[1,:])

    return Grid(cells, nodes, boundary=boundary, facesets=facesets)
end

# QuadraticQuadrilateral
function generate_grid{T}(::Type{QuadraticQuadrilateral}, nel::NTuple{2, Int}, left::Vec{2, T}=Vec{2}((-1.0,-1.0)), right::Vec{2, T}=Vec{2}((1.0,1.0)))
    nel_x = nel[1]; nel_y = nel[2]; nel_tot = nel_x*nel_y
    n_nodes_x = 2*nel_x + 1; n_nodes_y = 2*nel_y + 1
    n_nodes = n_nodes_x * n_nodes_y

    # Generate nodes
    coords_x = linspace(left[1], right[1], n_nodes_x)
    coords_y = linspace(left[2], right[2], n_nodes_y)
    nodes = Node{2,T}[]
    for j in 1:n_nodes_y, i in 1:n_nodes_x
        push!(nodes, Node((coords_x[i], coords_y[j])))
    end

    # Generate cells
    node_array = reshape(collect(1:n_nodes), (n_nodes_x, n_nodes_y))
    cells = QuadraticQuadrilateral[]
    for j in 1:nel_y, i in 1:nel_x
        push!(cells, QuadraticQuadrilateral((node_array[2*i-1,2*j-1],node_array[2*i+1,2*j-1],node_array[2*i+1,2*j+1],node_array[2*i-1,2*j+1],
                                             node_array[2*i,2*j-1],node_array[2*i+1,2*j],node_array[2*i,2*j+1],node_array[2*i-1,2*j],
                                             node_array[2*i,2*j])))
    end

    # Cell faces
    cell_array = reshape(collect(1:nel_tot),(nel_x, nel_y))
    boundary = CellFace[[CellFace((cl, 1)) for cl in cell_array[:,1]];
                              [CellFace((cl, 2)) for cl in cell_array[end,:]];
                              [CellFace((cl, 3)) for cl in cell_array[:,end]];
                              [CellFace((cl, 4)) for cl in cell_array[1,:]]]

    # Cell face sets
    offset = 0
    facesets = Dict{String, Vector{Int}}()
    facesets["bottom"] = (1:length(cell_array[:,1]))   + offset; offset += length(cell_array[:,1])
    facesets["right"]  = (1:length(cell_array[end,:])) + offset; offset += length(cell_array[end,:])
    facesets["top"]    = (1:length(cell_array[:,end])) + offset; offset += length(cell_array[:,end])
    facesets["left"]   = (1:length(cell_array[1,:]))   + offset; offset += length(cell_array[1,:])

    return Grid(cells, nodes, boundary=boundary, facesets=facesets)
end

# Hexahedron
function generate_grid{T}(::Type{Hexahedron}, nel::NTuple{3, Int}, left::Vec{3, T}=Vec{3}((-1.0,-1.0,-1.0)), right::Vec{3, T}=Vec{3}((1.0,1.0,1.0)))
    nel_x = nel[1]; nel_y = nel[2]; nel_z = nel[3]; nel_tot = nel_x*nel_y*nel_z
    n_nodes_x = nel_x + 1; n_nodes_y = nel_y + 1; n_nodes_z = nel_z + 1
    n_nodes = n_nodes_x * n_nodes_y * n_nodes_z

    # Generate nodes
    coords_x = linspace(left[1], right[1], n_nodes_x)
    coords_y = linspace(left[2], right[2], n_nodes_y)
    coords_z = linspace(left[3], right[3], n_nodes_z)
    nodes = Node{3,T}[]
    for k in 1:n_nodes_z, j in 1:n_nodes_y, i in 1:n_nodes_x
        push!(nodes, Node((coords_x[i], coords_y[j], coords_z[k])))
    end

    # Generate cells
    node_array = reshape(collect(1:n_nodes), (n_nodes_x, n_nodes_y, n_nodes_z))
    cells = Hexahedron[]
    for k in 1:nel_z, j in 1:nel_y, i in 1:nel_x
        push!(cells, Hexahedron((node_array[i,j,k], node_array[i+1,j,k], node_array[i+1,j+1,k], node_array[i,j+1,k],
                                 node_array[i,j,k+1], node_array[i+1,j,k+1], node_array[i+1,j+1,k+1], node_array[i,j+1,k+1])))
    end

    # Cell faces
    cell_array = reshape(collect(1:nel_tot),(nel_x, nel_y, nel_z))
    boundary = CellFace[[CellFace((cl, 1)) for cl in cell_array[:,:,1][:]];
                              [CellFace((cl, 2)) for cl in cell_array[:,1,:][:]];
                              [CellFace((cl, 3)) for cl in cell_array[end,:,:][:]];
                              [CellFace((cl, 4)) for cl in cell_array[:,end,:][:]];
                              [CellFace((cl, 5)) for cl in cell_array[1,:,:][:]];
                              [CellFace((cl, 6)) for cl in cell_array[:,:,end][:]]]

    # Cell face sets
    offset = 0
    facesets = Dict{String, Vector{Int}}()
    facesets["bottom"] = (1:length(cell_array[:,:,1][:]))   + offset; offset += length(cell_array[:,:,1][:])
    facesets["front"]  = (1:length(cell_array[:,1,:][:]))   + offset; offset += length(cell_array[:,1,:][:])
    facesets["right"]  = (1:length(cell_array[end,:,:][:])) + offset; offset += length(cell_array[end,:,:][:])
    facesets["back"]   = (1:length(cell_array[:,end,:][:])) + offset; offset += length(cell_array[:,end,:][:])
    facesets["left"]   = (1:length(cell_array[1,:,:][:]))   + offset; offset += length(cell_array[1,:,:][:])
    facesets["top"]    = (1:length(cell_array[:,:,end][:])) + offset; offset += length(cell_array[:,:,end][:])

    return Grid(cells, nodes, boundary=boundary, facesets=facesets)
end

# Triangle
function generate_grid{T}(::Type{Triangle}, nel::NTuple{2, Int}, left::Vec{2, T}=Vec{2}((-1.0,-1.0)), right::Vec{2, T}=Vec{2}((1.0,1.0)))
    nel_x = nel[1]; nel_y = nel[2]; nel_tot = 2*nel_x*nel_y
    n_nodes_x = nel_x + 1; n_nodes_y = nel_y + 1
    n_nodes = n_nodes_x * n_nodes_y

    # Generate nodes
    coords_x = linspace(left[1], right[1], n_nodes_x)
    coords_y = linspace(left[2], right[2], n_nodes_y)
    nodes = Node{2,T}[]
    for j in 1:n_nodes_y, i in 1:n_nodes_x
        push!(nodes, Node((coords_x[i], coords_y[j])))
    end

    # Generate cells
    node_array = reshape(collect(1:n_nodes), (n_nodes_x, n_nodes_y))
    cells = Triangle[]
    for j in 1:nel_y, i in 1:nel_x
        push!(cells, Triangle((node_array[i,j], node_array[i+1,j], node_array[i,j+1]))) # ◺
        push!(cells, Triangle((node_array[i+1,j], node_array[i+1,j+1], node_array[i,j+1]))) # ◹
    end

    # Cell faces
    cell_array = reshape(collect(1:nel_tot),(2, nel_x, nel_y))
    boundary = CellFace[[CellFace((cl, 1)) for cl in cell_array[1,:,1]];
                              [CellFace((cl, 1)) for cl in cell_array[2,end,:]];
                              [CellFace((cl, 2)) for cl in cell_array[2,:,end]];
                              [CellFace((cl, 3)) for cl in cell_array[1,1,:]]]

    # Cell face sets
    offset = 0
    facesets = Dict{String, Vector{Int}}()
    facesets["bottom"] = (1:length(cell_array[1,:,1]))   + offset; offset += length(cell_array[1,:,1])
    facesets["right"]  = (1:length(cell_array[2,end,:])) + offset; offset += length(cell_array[2,end,:])
    facesets["top"]    = (1:length(cell_array[2,:,end])) + offset; offset += length(cell_array[2,:,end])
    facesets["left"]   = (1:length(cell_array[1,1,:]))   + offset; offset += length(cell_array[1,1,:])

    return Grid(cells, nodes, boundary=boundary, facesets=facesets)
end

# QuadraticTriangle
function generate_grid{T}(::Type{QuadraticTriangle}, nel::NTuple{2, Int}, left::Vec{2, T}=Vec{2}((-1.0,-1.0)), right::Vec{2, T}=Vec{2}((1.0,1.0)))
    nel_x = nel[1]; nel_y = nel[2]; nel_tot = 2*nel_x*nel_y
    n_nodes_x = 2*nel_x + 1; n_nodes_y = 2*nel_y + 1
    n_nodes = n_nodes_x * n_nodes_y

    # Generate nodes
    coords_x = linspace(left[1], right[1], n_nodes_x)
    coords_y = linspace(left[2], right[2], n_nodes_y)
    nodes = Node{2,T}[]
    for j in 1:n_nodes_y, i in 1:n_nodes_x
        push!(nodes, Node((coords_x[i], coords_y[j])))
    end

    # Generate cells
    node_array = reshape(collect(1:n_nodes), (n_nodes_x, n_nodes_y))
    cells = QuadraticTriangle[]
    for j in 1:nel_y, i in 1:nel_x
        push!(cells, QuadraticTriangle((node_array[2*i-1,2*j-1], node_array[2*i+1,2*j-1], node_array[2*i-1,2*j+1],
                                        node_array[2*i,2*j-1], node_array[2*i,2*j], node_array[2*i-1,2*j]))) # ◺
        push!(cells, QuadraticTriangle((node_array[2*i+1,2*j-1], node_array[2*i+1,2*j+1], node_array[2*i-1,2*j+1],
                                        node_array[2*i+1,2*j], node_array[2*i,2*j+1], node_array[2*i,2*j]))) # ◹
    end

    # Cell faces
    cell_array = reshape(collect(1:nel_tot),(2, nel_x, nel_y))
    boundary = CellFace[[CellFace((cl, 1)) for cl in cell_array[1,:,1]];
                              [CellFace((cl, 1)) for cl in cell_array[2,end,:]];
                              [CellFace((cl, 2)) for cl in cell_array[2,:,end]];
                              [CellFace((cl, 3)) for cl in cell_array[1,1,:]]]

    # Cell face sets
    offset = 0
    facesets = Dict{String, Vector{Int}}()
    facesets["bottom"] = (1:length(cell_array[1,:,1]))   + offset; offset += length(cell_array[1,:,1])
    facesets["right"]  = (1:length(cell_array[2,end,:])) + offset; offset += length(cell_array[2,end,:])
    facesets["top"]    = (1:length(cell_array[2,:,end])) + offset; offset += length(cell_array[2,:,end])
    facesets["left"]   = (1:length(cell_array[1,1,:]))   + offset; offset += length(cell_array[1,1,:])

    return Grid(cells, nodes, boundary=boundary, facesets=facesets)
end

# Tetrahedron
function generate_grid{T}(::Type{Tetrahedron}, nel::NTuple{3, Int}, left::Vec{3, T}=Vec{3}((-1.0,-1.0,-1.0)), right::Vec{3, T}=Vec{3}((1.0,1.0,1.0)))
    nel_x = nel[1]; nel_y = nel[2]; nel_z = nel[3]; nel_tot = 5*nel_x*nel_y*nel_z
    n_nodes_x = nel_x + 1; n_nodes_y = nel_y + 1; n_nodes_z = nel_z + 1
    n_nodes = n_nodes_x * n_nodes_y * n_nodes_z

    # Generate nodes
    coords_x = linspace(left[1], right[1], n_nodes_x)
    coords_y = linspace(left[2], right[2], n_nodes_y)
    coords_z = linspace(left[3], right[3], n_nodes_z)
    nodes = Node{3,T}[]
    for k in 1:n_nodes_z, j in 1:n_nodes_y, i in 1:n_nodes_x
        push!(nodes, Node((coords_x[i], coords_y[j], coords_z[k])))
    end

    # Generate cells, case 13 from: http://www.baumanneduard.ch/Splitting%20a%20cube%20in%20tetrahedras2.htm
    node_array = reshape(collect(1:n_nodes), (n_nodes_x, n_nodes_y, n_nodes_z))
    cells = Tetrahedron[]
    for k in 1:nel_z, j in 1:nel_y, i in 1:nel_x
        tmp = (node_array[i,j,k], node_array[i+1,j,k], node_array[i+1,j+1,k], node_array[i,j+1,k],
               node_array[i,j,k+1], node_array[i+1,j,k+1], node_array[i+1,j+1,k+1], node_array[i,j+1,k+1])
        push!(cells, Tetrahedron((tmp[1], tmp[2], tmp[4], tmp[5])))
        push!(cells, Tetrahedron((tmp[2], tmp[3], tmp[4], tmp[7])))
        push!(cells, Tetrahedron((tmp[2], tmp[4], tmp[5], tmp[7])))
        push!(cells, Tetrahedron((tmp[2], tmp[5], tmp[6], tmp[7])))
        push!(cells, Tetrahedron((tmp[4], tmp[5], tmp[7], tmp[8])))
    end

    # Cell faces
    cell_array = reshape(collect(1:nel_tot),(5, nel_x, nel_y, nel_z))
    boundary = CellFace[[CellFace((cl, 1)) for cl in cell_array[1,:,:,1][:]];
                              [CellFace((cl, 1)) for cl in cell_array[2,:,:,1][:]];
                              [CellFace((cl, 2)) for cl in cell_array[1,:,1,:][:]];
                              [CellFace((cl, 1)) for cl in cell_array[4,:,1,:][:]];
                              [CellFace((cl, 2)) for cl in cell_array[2,end,:,:][:]];
                              [CellFace((cl, 4)) for cl in cell_array[4,end,:,:][:]];
                              [CellFace((cl, 3)) for cl in cell_array[2,:,end,:][:]];
                              [CellFace((cl, 4)) for cl in cell_array[5,:,end,:][:]];
                              [CellFace((cl, 4)) for cl in cell_array[1,1,:,:][:]];
                              [CellFace((cl, 2)) for cl in cell_array[5,1,:,:][:]];
                              [CellFace((cl, 3)) for cl in cell_array[4,:,:,end][:]];
                              [CellFace((cl, 3)) for cl in cell_array[5,:,:,end][:]]]

    # Cell face sets
    offset = 0
    facesets = Dict{String, Vector{Int}}()
    facesets["bottom"] = (1:length([cell_array[1,:,:,1][:];   cell_array[2,:,:,1][:]]))   + offset; offset += length([cell_array[1,:,:,1][:];   cell_array[2,:,:,1][:]])
    facesets["front"]  = (1:length([cell_array[1,:,1,:][:];   cell_array[4,:,1,:][:]]))   + offset; offset += length([cell_array[1,:,1,:][:];   cell_array[4,:,1,:][:]])
    facesets["right"]  = (1:length([cell_array[2,end,:,:][:]; cell_array[4,end,:,:][:]])) + offset; offset += length([cell_array[2,end,:,:][:]; cell_array[4,end,:,:][:]])
    facesets["back"]   = (1:length([cell_array[2,:,end,:][:]; cell_array[5,:,end,:][:]])) + offset; offset += length([cell_array[2,:,end,:][:]; cell_array[5,:,end,:][:]])
    facesets["left"]   = (1:length([cell_array[1,1,:,:][:];   cell_array[5,1,:,:][:]]))   + offset; offset += length([cell_array[1,1,:,:][:];   cell_array[5,1,:,:][:]])
    facesets["top"]    = (1:length([cell_array[4,:,:,end][:]; cell_array[5,:,:,end][:]])) + offset; offset += length([cell_array[4,:,:,end][:]; cell_array[5,:,:,end][:]])

    return Grid(cells, nodes, boundary=boundary, facesets=facesets)
end
