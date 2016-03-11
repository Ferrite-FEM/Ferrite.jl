# Typ definitions
type FEBoundary{T}
    topology::Array{Int}
end

type FEMesh{dim, T <: Real}
    coords::Vector{Vec{dim,T}} # Node coordinates
    topology::Matrix{Int} # Element nodes
    boundary::Vector{Vector{}} # Boundary nodes
end


function FEMesh() # Input should be shape, vectors with start and end-coordinates, and a vector with number of elements in each dir.
    # Set up the mesh here
    # Do some input checks and stuff

    # Here I was thinking we could call a set_up_mesh function that is overloaded for 1,2 and 3 dimensions somehow
    # coords,topology,boundary = set_up_mesh(dim)

    return FEMesh(coords,topology,boundary)
end


#############
# Accessors #
#############

# Should these even exist? Atleast I should add the same accessors for the FEDofs aswell so that contains all information needed

"""
    get_element_nodes(fe_m::FEMesh, elindex::Int) -> el_nodes::Vector
Gets the nodes for a given element index
"""
@inline get_element_nodes(fe_m::FEMesh, elindex::Int) = fe_m.topology[:,elindex]

"""
    get_element_coords(fe_m::FEMesh, elindex::Int) -> elcoord::Vector{Tensor{1,dim}}
Gets the coordinates of the nodes for a given element index
"""
@inline get_element_coords(fe_m::FEMesh, elindex::Int) = fe_m.coords[fe_m.topology[:,elindex]]

"""
    get_node_coords(fe_m::FEMesh, nodeindex::Int) -> nodecoord::Tensor{1,dim}
Gets the coordinates of the nodes for a given element index
"""
@inline get_node_coords(fe_m::FEMesh, nodeindex::Int) = fe_m.coords[nodeindex]

"""
    get_boundary_nodes(fe_m::FEMesh, boundaryindex::Int) -> boundarynodes::Vector
Gets the nodes on a specified boundary
"""
@inline get_boundary_nodes(fe_m::FEMesh, boundaryindex::Int) = fe_m.boundary[boundaryindex]

###

# Overloada set_up_mesh för olika dimensioner och former typ?
# 1D
function set_up_mesh_1D_square{T}(xs::Vector{T}, xe::Vector{T}, nel::Vector{Int}) # This is not quite finished yet I realize
    xs_x = xs[1]
    xe_x = xe[1]
    nel_x = nel[1]
    nel = nel_x
    n_nodes_x = nel_x + 1
    n_nodes = n_nodes_x

    coords = collect(linspace(xs_x, xe_x, n_nodes))
    tensor_coords = reinterpret(Vec{1, Float64}, coords, (length(coords),))

    nodes = collect(1:n_nodes)

    topology = zeros(Int,2,nel_x)

    for i = 1:nel_x
        topology[:,i] = nodes[i:i+1]
    end

    b0 = [FEBoundary(0,[1]), FEBoundary(0,[n_nodes])]

    boundary = Vector[b0]

    return FEMesh(tensor_coords,topology,boundary)
end



# 2D
function set_up_mesh_2D_square{T}(xs::Vector{T}, xe::Vector{T}, nel::Vector{Int})
    xs_x = xs[1]; xs_y = xs[2]
    xe_x = xe[1]; xe_y = xe[2]
    nel_x = nel[1]; nel_y = nel[2]
    nel = nel_x * nel_y
    n_nodes_x = nel_x + 1; n_nodes_y = nel_y + 1
    n_nodes = n_nodes_x * n_nodes_y

    coords_x = linspace(xs_x, xe_x, n_nodes_x)
    coords_y = linspace(xs_y, xe_y, n_nodes_y)

    coords = T[]
    for j in 1:n_nodes_y, i in 1:n_nodes_x
        ccoords = [coords_x[i], coords_y[j]]
        append!(coords,ccoords)
    end
    coords = reshape(coords,(2,n_nodes))

    tensor_coords = reinterpret(Vec{2, Float64}, coords, (size(coords,2),))

    nodes = reshape(collect(1:n_nodes),(n_nodes_x,n_nodes_y))

    topology = Int[]
    for j in 1:nel_y, i in 1:nel_x
        ctopology = [nodes[i:i+1,j]; nodes[i+1:-1:i,j+1]]
        append!(topology,ctopology)
    end
    topology = reshape(topology,(4,nel))

    # Edges
    b1D = [FEBoundary{1}(collect(1:n_nodes_x)),
           FEBoundary{1}(collect(n_nodes_x:n_nodes_x:n_nodes)),
           FEBoundary{1}(collect((n_nodes_x*(n_nodes_y-1)+1):n_nodes)),
           FEBoundary{1}(collect(1:n_nodes_x:(n_nodes_x*(n_nodes_y-1)+1)))]

    # Corners
    b0D = [FEBoundary{0}([1]),
           FEBoundary{0}([n_nodes_x]),
           FEBoundary{0}([n_nodes_x*(n_nodes_y-1)+1]),
           FEBoundary{0}([n_nodes])]

    boundary = Vector[b1D,b0D]

    return FEMesh(tensor_coords,topology,boundary)
end

# 3D
function set_up_mesh_3D_square{T}(xs::Vector{T}, xe::Vector{T}, nel::Vector{Int})
    xs_x = xs[1]; xs_y = xs[2]; xs_z = xs[3]
    xe_x = xe[1]; xe_y = xe[2]; xe_z = xe[3]
    nel_x = nel[1]; nel_y = nel[2]; nel_z = nel[3]
    nel = nel_x * nel_y * nel_z
    n_nodes_x = nel_x + 1; n_nodes_y = nel_y + 1; n_nodes_z = nel_z + 1
    n_nodes = n_nodes_x * n_nodes_y * n_nodes_z

    coords_x = linspace(xs_x, xe_x, n_nodes_x)
    coords_y = linspace(xs_y, xe_y, n_nodes_y)
    coords_z = linspace(xs_z, xe_z, n_nodes_z)

    coords = T[]
    for k in 1:n_nodes_z, j in 1:n_nodes_y, i in 1:n_nodes_x
        ccoords = [coords_x[i], coords_y[j], coords_z[k]]
        append!(coords,ccoords)
    end
    coords = reshape(coords,(3,n_nodes))

    tensor_coords = reinterpret(Vec{3, Float64}, coords, (size(coords,2),))

    # Set up topology
    nodes = reshape(collect(1:n_nodes),(n_nodes_x,n_nodes_y,n_nodes_z))

    topology = Int[]
    for k in 1:nel_z, j in 1:nel_y, i in 1:nel_x
        ctopology = [nodes[i:i+1,j,k]; nodes[i+1:-1:i,j+1,k];
                     nodes[i:i+1,j,k+1]; nodes[i+1:-1:i,j+1,k+1]]
        append!(topology,ctopology)
    end
    topology = reshape(topology,(8,nel))

    # TODO: Boundaries
    # Sides
    b2D = []
    # Edges
    b1D = [FEBoundary{1}(collect(1:n_nodes_x)),
           FEBoundary{1}(collect(n_nodes_x:n_nodes_x:n_nodes)),
           FEBoundary{1}(collect((n_nodes_x*(n_nodes_y-1)+1):n_nodes)),
           FEBoundary{1}(collect(1:n_nodes_x:(n_nodes_x*(n_nodes_y-1)+1)))]

    # Corners
    b0D = [FEBoundary{0}([1]),
           FEBoundary{0}([n_nodes_x]),
           FEBoundary{0}([n_nodes_x*(n_nodes_y-1)+1]),
           FEBoundary{0}([n_nodes])]

    boundary = Vector[b1D,b0D]

    return FEMesh(tensor_coords,topology,boundary)
end