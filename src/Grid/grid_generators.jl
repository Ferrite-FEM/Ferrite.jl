function boundaries_to_sparse(boundary)
    I, J, V = Int[], Int[], Bool[]
    for faceindex in boundary
        cell, face = faceindex
        push!(I, face)
        push!(J, cell)
        push!(V, true)
    end
    return sparse(I, J, V)
end

"""
    generate_grid(celltype::Cell, nel::NTuple, [left::Vec, right::Vec)

Return a `Grid` for a rectangle in 1, 2 or 3 dimensions. `celltype` defined the type of cells,
e.g. `Triangle` or `Hexahedron`. `nel` is a tuple of the number of elements in each direction.
`left` and `right` are optional endpoints of the domain. Defaults to -1 and 1 in all directions.
"""
generate_grid

# Line
function generate_grid(::Type{Line}, nel::NTuple{1,Int}, left::Vec{1,T}=Vec{1}((-1.0,)), right::Vec{1,T}=Vec{1}((1.0,))) where {T}
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
    boundary = Vector([(1, 1),
                       (nel_x, 2)])

    boundary_matrix = boundaries_to_sparse(boundary)

    # Cell face sets
    facesets = Dict("left"  => Set{Tuple{Int,Int}}([boundary[1]]),
                    "right" => Set{Tuple{Int,Int}}([boundary[2]]))
    return Grid(cells, nodes, facesets=facesets, boundary_matrix=boundary_matrix)
end

# QuadraticLine
function generate_grid(::Type{QuadraticLine}, nel::NTuple{1,Int}, left::Vec{1,T}=Vec{1}((-1.0,)), right::Vec{1,T}=Vec{1}((1.0,))) where {T}
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
    boundary = Tuple{Int,Int}[(1, 1),
                         (nel_x, 2)]

    boundary_matrix = boundaries_to_sparse(boundary)

    # Cell face sets
    facesets = Dict("left"  => Set{Tuple{Int,Int}}([boundary[1]]),
                    "right" => Set{Tuple{Int,Int}}([boundary[2]]))
    return Grid(cells, nodes, facesets=facesets, boundary_matrix=boundary_matrix)
end

function _generate_2d_nodes!(nodes, nx, ny, LL, LR, UR, UL)
      for i in 0:ny-1
        ratio_bounds = i / (ny-1)

        x0 = LL[1] * (1 - ratio_bounds) + ratio_bounds * UL[1]
        x1 = LR[1] * (1 - ratio_bounds) + ratio_bounds * UR[1]

        y0 = LL[2] * (1 - ratio_bounds) + ratio_bounds * UL[2]
        y1 = LR[2] * (1 - ratio_bounds) + ratio_bounds * UR[2]

        for j in 0:nx-1
            ratio = j / (nx-1)
            x = x0 * (1 - ratio) + ratio * x1
            y = y0 * (1 - ratio) + ratio * y1
            push!(nodes, Node((x, y)))
        end
    end
end


function generate_grid(C::Type{Cell{2,M,N}}, nel::NTuple{2,Int}, X::Vector{Vec{2,T}}) where {M,N,T}
    @assert length(X) == 4
    generate_grid(C, nel, X[1], X[2], X[3], X[4])
end

function generate_grid(C::Type{Cell{2,M,N}}, nel::NTuple{2,Int}, left::Vec{2,T}=Vec{2}((-1.0,-1.0)), right::Vec{2,T}=Vec{2}((1.0,1.0))) where {M,N,T}
    LL = left
    UR = right
    LR = Vec{2}((UR[1], LL[2]))
    UL = Vec{2}((LL[1], UR[2]))
    generate_grid(C, nel, LL, LR, UR, UL)
end

# Quadrilateral
function generate_grid(C::Type{Quadrilateral}, nel::NTuple{2,Int}, LL::Vec{2,T}, LR::Vec{2,T}, UR::Vec{2,T}, UL::Vec{2,T}) where {T}
    nel_x = nel[1]; nel_y = nel[2]; nel_tot = nel_x*nel_y
    n_nodes_x = nel_x + 1; n_nodes_y = nel_y + 1
    n_nodes = n_nodes_x * n_nodes_y

    # Generate nodes
    nodes = Node{2,T}[]
    _generate_2d_nodes!(nodes, n_nodes_x, n_nodes_y, LL, LR, UR, UL)

    # Generate cells
    node_array = reshape(collect(1:n_nodes), (n_nodes_x, n_nodes_y))
    cells = Quadrilateral[]
    for j in 1:nel_y, i in 1:nel_x
        push!(cells, Quadrilateral((node_array[i,j], node_array[i+1,j], node_array[i+1,j+1], node_array[i,j+1])))
    end

    # Cell faces
    cell_array = reshape(collect(1:nel_tot),(nel_x, nel_y))
    boundary = Tuple{Int,Int}[[(cl, 1) for cl in cell_array[:,1]];
                              [(cl, 2) for cl in cell_array[end,:]];
                              [(cl, 3) for cl in cell_array[:,end]];
                              [(cl, 4) for cl in cell_array[1,:]]]

    boundary_matrix = boundaries_to_sparse(boundary)

    # Cell face sets
    offset = 0
    facesets = Dict{String, Set{Tuple{Int,Int}}}()
    facesets["bottom"] = Set{Tuple{Int,Int}}(boundary[(1:length(cell_array[:,1]))   + offset]); offset += length(cell_array[:,1])
    facesets["right"]  = Set{Tuple{Int,Int}}(boundary[(1:length(cell_array[end,:])) + offset]); offset += length(cell_array[end,:])
    facesets["top"]    = Set{Tuple{Int,Int}}(boundary[(1:length(cell_array[:,end])) + offset]); offset += length(cell_array[:,end])
    facesets["left"]   = Set{Tuple{Int,Int}}(boundary[(1:length(cell_array[1,:]))   + offset]); offset += length(cell_array[1,:])

    return Grid(cells, nodes, facesets=facesets, boundary_matrix=boundary_matrix)
end

# QuadraticQuadrilateral
function generate_grid(::Type{QuadraticQuadrilateral}, nel::NTuple{2,Int}, LL::Vec{2,T}, LR::Vec{2,T}, UR::Vec{2,T}, UL::Vec{2,T}) where {T}
    nel_x = nel[1]; nel_y = nel[2]; nel_tot = nel_x*nel_y
    n_nodes_x = 2*nel_x + 1; n_nodes_y = 2*nel_y + 1
    n_nodes = n_nodes_x * n_nodes_y

    # Generate nodes
    nodes = Node{2,T}[]
    _generate_2d_nodes!(nodes, n_nodes_x, n_nodes_y, LL, LR, UR, UL)

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
    boundary = Tuple{Int,Int}[[(cl, 1) for cl in cell_array[:,1]];
                              [(cl, 2) for cl in cell_array[end,:]];
                              [(cl, 3) for cl in cell_array[:,end]];
                              [(cl, 4) for cl in cell_array[1,:]]]

    boundary_matrix = boundaries_to_sparse(boundary)

    # Cell face sets
    offset = 0
    facesets = Dict{String, Set{Tuple{Int,Int}}}()
    facesets["bottom"] = Set{Tuple{Int,Int}}(boundary[(1:length(cell_array[:,1]))   + offset]); offset += length(cell_array[:,1])
    facesets["right"]  = Set{Tuple{Int,Int}}(boundary[(1:length(cell_array[end,:])) + offset]); offset += length(cell_array[end,:])
    facesets["top"]    = Set{Tuple{Int,Int}}(boundary[(1:length(cell_array[:,end])) + offset]); offset += length(cell_array[:,end])
    facesets["left"]   = Set{Tuple{Int,Int}}(boundary[(1:length(cell_array[1,:]))   + offset]); offset += length(cell_array[1,:])

    return Grid(cells, nodes, facesets=facesets, boundary_matrix=boundary_matrix)
end

# Hexahedron
function generate_grid(::Type{Hexahedron}, nel::NTuple{3,Int}, left::Vec{3,T}=Vec{3}((-1.0,-1.0,-1.0)), right::Vec{3,T}=Vec{3}((1.0,1.0,1.0))) where {T}
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
    boundary = Tuple{Int,Int}[[(cl, 1) for cl in cell_array[:,:,1][:]];
                              [(cl, 2) for cl in cell_array[:,1,:][:]];
                              [(cl, 3) for cl in cell_array[end,:,:][:]];
                              [(cl, 4) for cl in cell_array[:,end,:][:]];
                              [(cl, 5) for cl in cell_array[1,:,:][:]];
                              [(cl, 6) for cl in cell_array[:,:,end][:]]]

    boundary_matrix = boundaries_to_sparse(boundary)

    # Cell face sets
    offset = 0
    facesets = Dict{String,Set{Tuple{Int,Int}}}()
    facesets["bottom"] = Set{Tuple{Int,Int}}(boundary[(1:length(cell_array[:,:,1][:]))   + offset]); offset += length(cell_array[:,:,1][:])
    facesets["front"]  = Set{Tuple{Int,Int}}(boundary[(1:length(cell_array[:,1,:][:]))   + offset]); offset += length(cell_array[:,1,:][:])
    facesets["right"]  = Set{Tuple{Int,Int}}(boundary[(1:length(cell_array[end,:,:][:])) + offset]); offset += length(cell_array[end,:,:][:])
    facesets["back"]   = Set{Tuple{Int,Int}}(boundary[(1:length(cell_array[:,end,:][:])) + offset]); offset += length(cell_array[:,end,:][:])
    facesets["left"]   = Set{Tuple{Int,Int}}(boundary[(1:length(cell_array[1,:,:][:]))   + offset]); offset += length(cell_array[1,:,:][:])
    facesets["top"]    = Set{Tuple{Int,Int}}(boundary[(1:length(cell_array[:,:,end][:])) + offset]); offset += length(cell_array[:,:,end][:])

    return Grid(cells, nodes, facesets=facesets, boundary_matrix=boundary_matrix)
end

# Triangle
function generate_grid(::Type{Triangle}, nel::NTuple{2,Int}, LL::Vec{2,T}, LR::Vec{2,T}, UR::Vec{2,T}, UL::Vec{2,T}) where {T}
    nel_x = nel[1]; nel_y = nel[2]; nel_tot = 2*nel_x*nel_y
    n_nodes_x = nel_x + 1; n_nodes_y = nel_y + 1
    n_nodes = n_nodes_x * n_nodes_y

    # Generate nodes
    nodes = Node{2,T}[]
    _generate_2d_nodes!(nodes, n_nodes_x, n_nodes_y, LL, LR, UR, UL)

    # Generate cells
    node_array = reshape(collect(1:n_nodes), (n_nodes_x, n_nodes_y))
    cells = Triangle[]
    for j in 1:nel_y, i in 1:nel_x
        push!(cells, Triangle((node_array[i,j], node_array[i+1,j], node_array[i,j+1]))) # ◺
        push!(cells, Triangle((node_array[i+1,j], node_array[i+1,j+1], node_array[i,j+1]))) # ◹
    end

    # Cell faces
    cell_array = reshape(collect(1:nel_tot),(2, nel_x, nel_y))
    boundary = Tuple{Int,Int}[[(cl, 1) for cl in cell_array[1,:,1]];
                               [(cl, 1) for cl in cell_array[2,end,:]];
                               [(cl, 2) for cl in cell_array[2,:,end]];
                               [(cl, 3) for cl in cell_array[1,1,:]]]

    boundary_matrix = boundaries_to_sparse(boundary)

    # Cell face sets
    offset = 0
    facesets = Dict{String,Set{Tuple{Int,Int}}}()
    facesets["bottom"] = Set{Tuple{Int,Int}}(boundary[(1:length(cell_array[1,:,1]))   + offset]); offset += length(cell_array[1,:,1])
    facesets["right"]  = Set{Tuple{Int,Int}}(boundary[(1:length(cell_array[2,end,:])) + offset]); offset += length(cell_array[2,end,:])
    facesets["top"]    = Set{Tuple{Int,Int}}(boundary[(1:length(cell_array[2,:,end])) + offset]); offset += length(cell_array[2,:,end])
    facesets["left"]   = Set{Tuple{Int,Int}}(boundary[(1:length(cell_array[1,1,:]))   + offset]); offset += length(cell_array[1,1,:])

    return Grid(cells, nodes, facesets=facesets, boundary_matrix=boundary_matrix)
end

# QuadraticTriangle
function generate_grid(::Type{QuadraticTriangle}, nel::NTuple{2,Int}, LL::Vec{2,T}, LR::Vec{2,T}, UR::Vec{2,T}, UL::Vec{2,T}) where {T}
    nel_x = nel[1]; nel_y = nel[2]; nel_tot = 2*nel_x*nel_y
    n_nodes_x = 2*nel_x + 1; n_nodes_y = 2*nel_y + 1
    n_nodes = n_nodes_x * n_nodes_y

    # Generate nodes
    nodes = Node{2,T}[]
    _generate_2d_nodes!(nodes, n_nodes_x, n_nodes_y, LL, LR, UR, UL)

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
    boundary = Tuple{Int,Int}[[(cl, 1) for cl in cell_array[1,:,1]];
                              [(cl, 1) for cl in cell_array[2,end,:]];
                              [(cl, 2) for cl in cell_array[2,:,end]];
                              [(cl, 3) for cl in cell_array[1,1,:]]]

    boundary_matrix = boundaries_to_sparse(boundary)

    # Cell face sets
    offset = 0
    facesets = Dict{String,Set{Tuple{Int,Int}}}()
    facesets["bottom"] = Set{Tuple{Int,Int}}(boundary[(1:length(cell_array[1,:,1]))   + offset]); offset += length(cell_array[1,:,1])
    facesets["right"]  = Set{Tuple{Int,Int}}(boundary[(1:length(cell_array[2,end,:])) + offset]); offset += length(cell_array[2,end,:])
    facesets["top"]    = Set{Tuple{Int,Int}}(boundary[(1:length(cell_array[2,:,end])) + offset]); offset += length(cell_array[2,:,end])
    facesets["left"]   = Set{Tuple{Int,Int}}(boundary[(1:length(cell_array[1,1,:]))   + offset]); offset += length(cell_array[1,1,:])

    return Grid(cells, nodes, facesets=facesets, boundary_matrix=boundary_matrix)
end

# Tetrahedron
function generate_grid(::Type{Tetrahedron}, nel::NTuple{3,Int}, left::Vec{3,T}=Vec{3}((-1.0,-1.0,-1.0)), right::Vec{3,T}=Vec{3}((1.0,1.0,1.0))) where {T}
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
    boundary = Tuple{Int,Int}[[(cl, 1) for cl in cell_array[1,:,:,1][:]];
                              [(cl, 1) for cl in cell_array[2,:,:,1][:]];
                              [(cl, 2) for cl in cell_array[1,:,1,:][:]];
                              [(cl, 1) for cl in cell_array[4,:,1,:][:]];
                              [(cl, 2) for cl in cell_array[2,end,:,:][:]];
                              [(cl, 4) for cl in cell_array[4,end,:,:][:]];
                              [(cl, 3) for cl in cell_array[2,:,end,:][:]];
                              [(cl, 4) for cl in cell_array[5,:,end,:][:]];
                              [(cl, 4) for cl in cell_array[1,1,:,:][:]];
                              [(cl, 2) for cl in cell_array[5,1,:,:][:]];
                              [(cl, 3) for cl in cell_array[4,:,:,end][:]];
                              [(cl, 3) for cl in cell_array[5,:,:,end][:]]]

    boundary_matrix = boundaries_to_sparse(boundary)

    # Cell face sets
    offset = 0
    facesets = Dict{String, Set{Tuple{Int,Int}}}()
    facesets["bottom"] = Set{Tuple{Int,Int}}(boundary[(1:length([cell_array[1,:,:,1][:];   cell_array[2,:,:,1][:]]))   + offset]); offset += length([cell_array[1,:,:,1][:];   cell_array[2,:,:,1][:]])
    facesets["front"]  = Set{Tuple{Int,Int}}(boundary[(1:length([cell_array[1,:,1,:][:];   cell_array[4,:,1,:][:]]))   + offset]); offset += length([cell_array[1,:,1,:][:];   cell_array[4,:,1,:][:]])
    facesets["right"]  = Set{Tuple{Int,Int}}(boundary[(1:length([cell_array[2,end,:,:][:]; cell_array[4,end,:,:][:]])) + offset]); offset += length([cell_array[2,end,:,:][:]; cell_array[4,end,:,:][:]])
    facesets["back"]   = Set{Tuple{Int,Int}}(boundary[(1:length([cell_array[2,:,end,:][:]; cell_array[5,:,end,:][:]])) + offset]); offset += length([cell_array[2,:,end,:][:]; cell_array[5,:,end,:][:]])
    facesets["left"]   = Set{Tuple{Int,Int}}(boundary[(1:length([cell_array[1,1,:,:][:];   cell_array[5,1,:,:][:]]))   + offset]); offset += length([cell_array[1,1,:,:][:];   cell_array[5,1,:,:][:]])
    facesets["top"]    = Set{Tuple{Int,Int}}(boundary[(1:length([cell_array[4,:,:,end][:]; cell_array[5,:,:,end][:]])) + offset]); offset += length([cell_array[4,:,:,end][:]; cell_array[5,:,:,end][:]])

    return Grid(cells, nodes, facesets=facesets, boundary_matrix=boundary_matrix)
end
