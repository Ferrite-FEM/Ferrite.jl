# Typ definitions
type FEBoundary # Tänkte typ parametrisera detta på nåt sätt så att man har FEboundary i 2D, 1D, och 0D
    dim::Int
    topology::Array{Int}
end

type FEMesh{dim, T <: Real}
    coords::Vector{Vec{dim,T}} # Node coordinates
    topology::Matrix{Int} # Element nodes
    boundary::Vector{Vector{}} # Boundary nodes
end


function FEMesh(dim)
    # Set up the mesh here

    # Do some input checks and stuff

    # Here I was thinking we could call a set_up_mesh function that is overloaded for 1,2 and 3 dimensions somehow
    coords,topology,boundary = set_up_mesh(dim)

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
function set_up_mesh{T}(xs::Vector{T}, xe::Vector{T}, nel::Vector{Int}) # This is not quite finished yet I realize
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
function set_up_mesh{T}(xs::Vector{T}, xe::Vector{T}, nel::Vector{Int})
    xs_x = xs[1]; xs_y = xs[2]
    xe_x = xe[1]; xe_y = xe[2]
    nel_x = nel[1]; nel_y = nel[2]
    nel = nel_x * nel_y
    n_nodes_x = nel_x + 1; n_nodes_y = nel_y + 1
    n_nodes = n_nodes_x * n_nodes_y

    coords_x = repmat(collect(linspace(xs_x, xe_x, n_nodes_x)), n_nodes_y)
    coords_y = repmat(collect(linspace(xs_x, xe_x, n_nodes_y))', n_nodes_x)[:]
    coords = [coords_x';
              coords_y']
    tensor_coords = reinterpret(Vec{2, Float64}, coords, (size(coords,2),))

    nodes = reshape(collect(1:n_nodes),(n_nodes_x,n_nodes_y))


    cel = 1

    for i = 1:nEly, j = 1:nElx
        topology[:,cEl] = [(n_nodes_x*(i-1)+j):(n_nodes_x*(i-1)+j+1); flipdim(((n_nodes_x*(i-1)+j):(nxNodes*(i-1)+j+1))+nxNodes,1)]
        cEl += 1
    end

    topology = zeros(Int,2,nel_x)

    for i = 1:nel_x
        topology[:,i] = nodes[i:i+1]
    end

    b0 = [FEBoundary(0,[1]), FEBoundary(0,[n_nodes])]

    boundary = Vector[b0]

    return FEMesh(tensor_coords,topology,boundary)
end

# # 3D
# function set_up_mesh{T}(xs::Vector{T}, xe::Vector{T}, nel::Vector{Int})
#     xs_x = xs[1]; xs_y = xs[2]
#     xe_x = xe[1]; xe_y = xe[2]
#     nel_x = nel[1]; nel_y = nel[2]
#     nel = nel_x * nel_y
#     n_nodes_x = nel_x + 1; n_nodes_y = nel_y + 1
#     n_nodes = n_nodes_x * n_nodes_y

#     coords_x = repmat(collect(linspace(xs_x, xe_x, n_nodes_x)), n_nodes_y)
#     coords_y = repmat(collect(linspace(xs_x, xe_x, n_nodes_y))', n_nodes_x)[:]
#     coords = [coords_x';
#               coords_y']
#     tensor_coords = reinterpret(Vec{2, Float64}, coords, (size(coords,2),))

#     nodes = reshape(collect(1:n_nodes),(n_nodes_x,n_nodes_y))


#     cel = 1

#     for i = 1:nEly, j = 1:nElx
#         topology[:,cEl] = [(n_nodes_x*(i-1)+j):(n_nodes_x*(i-1)+j+1); flipdim(((n_nodes_x*(i-1)+j):(nxNodes*(i-1)+j+1))+nxNodes,1)]
#         cEl += 1
#     end

#     topology = zeros(Int,2,nel_x)

#     for i = 1:nel_x
#         topology[:,i] = nodes[i:i+1]
#     end

#     b0 = [FEBoundary(0,[1]), FEBoundary(0,[n_nodes])]

#     boundary = Vector[b0]

#     return FEMesh(tensor_coords,topology,boundary)
# end