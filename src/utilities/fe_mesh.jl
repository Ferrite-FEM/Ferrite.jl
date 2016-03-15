# Typ definitions
type FEBoundary{T}
    topology::Array{Int}
end

type FEMesh{dim, T <: Real}
    coords::Vector{Vec{dim,T}} # Node coordinates
    topology::Matrix{Int} # Element nodes
    boundary::Vector{Vector{}} # Boundary nodes
end


function FEMesh{T}(dim,shape::Shape,xs::Vector{T},xe::Vector{T},nel::Vector{Int}) # Input should be shape, vectors with start and end-coordinates, and a vector with number of elements in each dir.

    # Do some input checks and stuff here


    fe_m = generate_mesh(dim,shape,xs,xe,nel)

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

"""
    get_number_of_elements(fe_m::FEMesh) -> numel::Int
    get_number_of_elements(fe_b::FEBoundary) -> numel::Int
Gets the number of elements in a given FEMesh or FEBoundary
"""
@inline get_number_of_elements(fe_m::FEMesh) = size(fe_m.topology,2)

@inline get_number_of_elements{T}(fe_b::FEBoundary{T}) = size(fe_b.topology,2)



###################################################
# Mesh generating functions for simple geometries #
###################################################

################
# Square dim 1 #
################
function generate_mesh{T}(::Type{Dim{1}},::Square, xs::Vector{T}, xe::Vector{T}, nel::Vector{Int})

    xs_x = xs[1]
    xe_x = xe[1]
    nel_x = nel[1]
    nel = nel_x
    n_nodes_x = nel_x + 1
    n_nodes = n_nodes_x

    coords = collect(linspace(xs_x, xe_x, n_nodes))
    tensor_coords = reinterpret(Vec{1, Float64}, coords, (length(coords),))

    nodes = collect(1:n_nodes)

    topology = Int[]
    for i in 1:nel_x
        ctopology = nodes[i:i+1]
        append!(topology,ctopology)
    end
    topology = reshape(topology,(2,nel))

    b0 = [FEBoundary{0}([1]),
          FEBoundary{0}([n_nodes])]

    boundary = Vector[b0]

    return FEMesh(tensor_coords,topology,boundary)
end



################
# Square dim 2 #
################
function generate_mesh{T}(::Type{Dim{2}},::Square, xs::Vector{T}, xe::Vector{T}, nel::Vector{Int})

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
    edge_topology = Vector{Int}[
                    nodes[:,1][:], nodes[end,:][:], nodes[:,end][:], nodes[1,:][:]]

    b1D = FEBoundary{1}[]

    for edge in 1:length(edge_topology)
        nel_bound = length(edge_topology[edge])-1
        b1Dtopology = Int[]
        for i in 1:nel_bound
            append!(b1Dtopology,edge_topology[edge][i:i+1])
        end
        b1Dtopology = reshape(b1Dtopology,(2,nel_bound))
        push!(b1D,FEBoundary{1}(b1Dtopology))
    end

    # Corners
    b0D = [FEBoundary{0}([nodes[1,1]]),
           FEBoundary{0}([nodes[end,1]]),
           FEBoundary{0}([nodes[1,end]]),
           FEBoundary{0}([nodes[end,end]])]


    boundary = Vector[b1D,b0D]

    return FEMesh(tensor_coords,topology,boundary)
end



################
# Square dim 3 #
################
function generate_mesh{T}(::Type{Dim{3}},::Square, xs::Vector{T}, xe::Vector{T}, nel::Vector{Int})

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

    # Sides
    # REMARK: This looks really ugly, but wont be necessary on Julia 0.5 I think. Same with the vector operators on the edge_topology.
    side_topology = Matrix{Int}[
                    nodes[:,:,1],
                    reshape(nodes[:,1,:],size(nodes[:,1,:])[[1,3]]),
                    reshape(nodes[end,:,:],size(nodes[end,:,:])[[2,3]]),
                    reshape(nodes[:,end,:],size(nodes[:,end,:])[[1,3]]),
                    reshape(nodes[1,:,:],size(nodes[1,:,:])[[2,3]]),
                    nodes[:,:,end]]

    b2D = FEBoundary{2}[]

    for side in 1:length(side_topology)
        nel_boundx, nel_boundy = size(side_topology[side])
        nel_boundx -= 1; nel_boundy -= 1;

        b2Dtopology = Int[]
        for j in 1:nel_boundy, i in 1:nel_boundx
            cboundtop = [side_topology[side][i:i+1,j]; side_topology[side][i+1:-1:i,j+1]]
            append!(b2Dtopology,cboundtop)
        end
        b2Dtopology = reshape(b2Dtopology,(4,nel_boundx*nel_boundy))
        push!(b2D,FEBoundary{2}(b2Dtopology))
    end

    # Edges
    edge_topology = Vector{Int}[
                    nodes[:,1,1][:], nodes[end,:,1][:], nodes[:,end,1][:], nodes[1,:,1][:],
                    nodes[1,1,:][:], nodes[end,1,:][:], nodes[end,end,:][:], nodes[1,end,:][:],
                    nodes[:,1,end][:], nodes[end,:,end][:], nodes[:,end,end][:], nodes[1,:,end][:]]
    b1D = FEBoundary{1}[]

    for edge in 1:length(edge_topology)
        nel_bound = length(edge_topology[edge])-1
        b1Dtopology = Int[]
        for i in 1:nel_bound
            append!(b1Dtopology,edge_topology[edge][i:i+1])
        end
        b1Dtopology = reshape(b1Dtopology,(2,nel_bound))
        push!(b1D,FEBoundary{1}(b1Dtopology))
    end

    # Corners
    b0D = [FEBoundary{0}([nodes[1,1,1]]),
           FEBoundary{0}([nodes[end,1,1]]),
           FEBoundary{0}([nodes[end,end,1]]),
           FEBoundary{0}([nodes[1,end,1]]),
           FEBoundary{0}([nodes[1,1,end]]),
           FEBoundary{0}([nodes[end,1,end]]),
           FEBoundary{0}([nodes[end,end,end]]),
           FEBoundary{0}([nodes[1,end,end]])]

    boundary = Vector[b1D,b0D]

    return FEMesh(tensor_coords,topology,boundary)
end



##################
# Triangle dim 2 #
##################
function generate_mesh{T}(::Type{Dim{2}},::Triangle, xs::Vector{T}, xe::Vector{T}, nel::Vector{Int})

    xs_x = xs[1]; xs_y = xs[2]
    xe_x = xe[1]; xe_y = xe[2]
    nel_x = nel[1]; nel_y = nel[2]
    nel = 2 * nel_x * nel_y
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
        ctopology = [nodes[i:i+1,j]; nodes[i,j+1]]
        append!(topology,ctopology)
        ctopology = [nodes[i+1,j]; nodes[i+1:-1:i,j+1]]
        append!(topology,ctopology)
    end
    topology = reshape(topology,(3,nel))

    # Edges
    edge_topology = Vector{Int}[
                    nodes[:,1][:], nodes[end,:][:], nodes[:,end][:], nodes[1,:][:]]

    b1D = FEBoundary{1}[]

    for edge in 1:length(edge_topology)
        nel_bound = length(edge_topology[edge])-1
        b1Dtopology = Int[]
        for i in 1:nel_bound
            append!(b1Dtopology,edge_topology[edge][i:i+1])
        end
        b1Dtopology = reshape(b1Dtopology,(2,nel_bound))
        push!(b1D,FEBoundary{1}(b1Dtopology))
    end

    # Corners
    b0D = [FEBoundary{0}([nodes[1,1]]),
           FEBoundary{0}([nodes[end,1]]),
           FEBoundary{0}([nodes[1,end]]),
           FEBoundary{0}([nodes[end,end]])]

    boundary = Vector[b1D,b0D]

    return FEMesh(tensor_coords,topology,boundary)
end



##################
# Triangle dim 3 #
##################
function generate_mesh{T}(::Type{Dim{3}},::Triangle, xs::Vector{T}, xe::Vector{T}, nel::Vector{Int})

    xs_x = xs[1]; xs_y = xs[2]; xs_z = xs[3]
    xe_x = xe[1]; xe_y = xe[2]; xe_z = xe[3]
    nel_x = nel[1]; nel_y = nel[2]; nel_z = nel[3]
    nel = 6 * nel_x * nel_y * nel_z
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
    # Using # 1 from http://www.baumanneduard.ch/Splitting%20a%20cube%20in%20tetrahedras2.htm
    for k in 1:nel_z, j in 1:nel_y, i in 1:nel_x # Make sure numbering is ok with func_space
        ctopology = [nodes[i,j,k], nodes[i+1,j,k], nodes[i,j+1,k], nodes[i,j+1,k+1]]
        append!(topology,ctopology)
        ctopology = [nodes[i,j,k], nodes[i+1,j,k], nodes[i,j,k+1], nodes[i,j+1,k+1]]
        append!(topology,ctopology)
        ctopology = [nodes[i+1,j,k], nodes[i+1,j+1,k], nodes[i,j+1,k], nodes[i,j+1,k+1]]
        append!(topology,ctopology)
        ctopology = [nodes[i+1,j,k], nodes[i+1,j+1,k], nodes[i+1,j+1,k+1], nodes[i,j+1,k+1]]
        append!(topology,ctopology)
        ctopology = [nodes[i+1,j,k], nodes[i,j,k+1], nodes[i+1,j,k+1], nodes[i,j+1,k+1]]
        append!(topology,ctopology)
        ctopology = [nodes[i+1,j,k], nodes[i+1,j,k+1], nodes[i+1,j+1,k+1], nodes[i,j+1,k+1]]
        append!(topology,ctopology)
    end
    topology = reshape(topology,(4,nel))

    # Sides
    side_topology = Matrix{Int}[
                    nodes[:,:,1],
                    reshape(nodes[:,1,:],size(nodes[:,1,:])[[1,3]]),
                    reshape(nodes[end,:,:],size(nodes[end,:,:])[[2,3]]),
                    reshape(nodes[:,end,:],size(nodes[:,end,:])[[1,3]]),
                    reshape(nodes[1,:,:],size(nodes[1,:,:])[[2,3]]),
                    nodes[:,:,end]]


    b2D = FEBoundary{2}[]

    for side in 1:length(side_topology)
        nel_boundx, nel_boundy = size(side_topology[side])
        nel_boundx -= 1; nel_boundy -= 1;

        b2Dtopology = Int[]
        for j in 1:nel_boundy, i in 1:nel_boundx # This might be a problem if the actual element sides is needed, but should work just fine.
            cboundtop = [side_topology[side][i:i+1,j]; side_topology[side][i,j+1]]
            append!(b2Dtopology,cboundtop)
            cboundtop = [side_topology[side][i+1,j]; side_topology[side][i+1:-1:i,j+1]]
            append!(b2Dtopology,cboundtop)
        end
        b2Dtopology = reshape(b2Dtopology,(3,2*nel_boundx*nel_boundy))
        push!(b2D,FEBoundary{2}(b2Dtopology))
    end

    # Edges
    edge_topology = Vector{Int}[
                    nodes[:,1,1][:], nodes[end,:,1][:], nodes[:,end,1][:], nodes[1,:,1][:],
                    nodes[1,1,:][:], nodes[end,1,:][:], nodes[end,end,:][:], nodes[1,end,:][:],
                    nodes[:,1,end][:], nodes[end,:,end][:], nodes[:,end,end][:], nodes[1,:,end][:]]
    b1D = FEBoundary{1}[]

    for edge in 1:length(edge_topology)
        nel_bound = length(edge_topology[edge])-1
        b1Dtopology = Int[]
        for i in 1:nel_bound
            append!(b1Dtopology,edge_topology[edge][i:i+1])
        end
        b1Dtopology = reshape(b1Dtopology,(2,nel_bound))
        push!(b1D,FEBoundary{1}(b1Dtopology))
    end

    # Corners
    b0D = [FEBoundary{0}([nodes[1,1,1]]),
           FEBoundary{0}([nodes[end,1,1]]),
           FEBoundary{0}([nodes[end,end,1]]),
           FEBoundary{0}([nodes[1,end,1]]),
           FEBoundary{0}([nodes[1,1,end]]),
           FEBoundary{0}([nodes[end,1,end]]),
           FEBoundary{0}([nodes[end,end,end]]),
           FEBoundary{0}([nodes[1,end,end]])]

    boundary = Vector[b1D,b0D]

    return FEMesh(tensor_coords,topology,boundary)
end