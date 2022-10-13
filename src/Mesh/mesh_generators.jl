"""
    generate_mesh(elementtype::Element, nel::NTuple, [left::Vec, right::Vec)

Return a `Mesh` for a rectangle in 1, 2 or 3 dimensions. `elementtype` defined the type of elements,
e.g. `Triangle` or `Hexahedron`. `nel` is a tuple of the number of elements in each direction.
`left` and `right` are optional endpoints of the domain. Defaults to -1 and 1 in all directions.
"""
generate_mesh

# Helper to generate geometry in unit cubes
function generate_mesh(C::Type{Element{Dim,Ref,N}}, nel::NTuple{Dim,Int}, SDim::Int) where {Dim,Ref,N}
    generate_mesh(C, nel, Vec{SDim}(ntuple(x->-1.0, SDim)), Vec{SDim}(ntuple(x->1.0, SDim)))
end

function generate_mesh(C::Type{Element{2,Ref,N}}, nel::NTuple{2,Int}, left::Vec{2,T}=Vec{2}((-1.0,-1.0)), right::Vec{2,T}=Vec{2}((1.0,1.0))) where {Ref,N,T}
    LL = left
    UR = right
    LR = Vec{2}((UR[1], LL[2]))
    UL = Vec{2}((LL[1], UR[2]))
    generate_mesh(C, nel, LL, LR, UR, UL)
end

# Line
function generate_mesh(::Type{LineElement}, nel::NTuple{1,Int}, left::Vec{Dim,T}=Vec{1}((1.0,)), right::Vec{Dim,T}=Vec{1}((1.0,))) where {Dim, T}
    nel_x = nel[1]
    n_nodes = nel_x + 1

    # Generate nodes
    nodes = Node{Dim,T}[]
    for coord in range(0.0, 1.0, length=n_nodes)
        push!(nodes, Node(left*coord + right*(1.0-coord)))
    end

    # Generate elements
    elements = LineElement[]
    for i in 1:nel_x
        push!(elements, LineElement((i, i+1)))
    end

    # Element faces
    boundary = Vector([FaceIndex(1, 1),
                       FaceIndex(nel_x, 2)])

    # Element face sets
    facesets = Dict("left"  => Set{FaceIndex}([boundary[1]]),
                    "right" => Set{FaceIndex}([boundary[2]]))
    return Mesh(elements, nodes; facesets=facesets)
end

# QuadraticLine
function generate_mesh(::Type{QuadraticLineElement}, nel::NTuple{1,Int}, left::Vec{Dim,T}=Vec{1}(ntuple(x->-Int(x==1), 1)), right::Vec{Dim,T}=Vec{1}(ntuple(x->-Int(x==1), 1))) where {Dim, T}
    nel_x = nel[1]
    n_nodes = 2*nel_x + 1

    # Generate n_nodes
    nodes = Node{Dim,T}[]
    for coord in range(0.0, 1.0, length=n_nodes)
        push!(nodes, left*coord + right*(1.0-coord))
    end

    # Generate elements
    elements = QuadraticLineElement[]
    for i in 1:nel_x
        push!(elements, QuadraticLineElement((2*i-1, 2*i+1, 2*i)))
    end

    # Element faces
    boundary = FaceIndex[FaceIndex(1, 1),
                         FaceIndex(nel_x, 2)]

    # Element face sets
    facesets = Dict("left"  => Set{FaceIndex}([boundary[1]]),
                    "right" => Set{FaceIndex}([boundary[2]]))
    return Mesh(elements, nodes; facesets=facesets)
end

# Quadrilateral
function generate_mesh(C::Type{QuadrilateralElement}, nel::NTuple{2,Int}, LL::Vec{2,T}, LR::Vec{2,T}, UR::Vec{2,T}, UL::Vec{2,T}) where {T}
    nel_x = nel[1]; nel_y = nel[2]; nel_tot = nel_x*nel_y
    n_nodes_x = nel_x + 1; n_nodes_y = nel_y + 1
    n_nodes = n_nodes_x * n_nodes_y

    # Generate nodes
    nodes = Node{2,T}[]
    _generate_2d_nodes!(nodes, n_nodes_x, n_nodes_y, LL, LR, UR, UL)

    # Generate elements
    node_array = reshape(collect(1:n_nodes), (n_nodes_x, n_nodes_y))
    elements = QuadrilateralElement[]
    for j in 1:nel_y, i in 1:nel_x
        push!(elements, QuadrilateralElement((node_array[i,j], node_array[i+1,j], node_array[i+1,j+1], node_array[i,j+1])))
    end

    # Element faces
    element_array = reshape(collect(1:nel_tot),(nel_x, nel_y))
    boundary = FaceIndex[[FaceIndex(cl, 1) for cl in element_array[:,1]];
                              [FaceIndex(cl, 2) for cl in element_array[end,:]];
                              [FaceIndex(cl, 3) for cl in element_array[:,end]];
                              [FaceIndex(cl, 4) for cl in element_array[1,:]]]

    # Element face sets
    offset = 0
    facesets = Dict{String, Set{FaceIndex}}()
    facesets["bottom"] = Set{FaceIndex}(boundary[(1:length(element_array[:,1]))   .+ offset]); offset += length(element_array[:,1])
    facesets["right"]  = Set{FaceIndex}(boundary[(1:length(element_array[end,:])) .+ offset]); offset += length(element_array[end,:])
    facesets["top"]    = Set{FaceIndex}(boundary[(1:length(element_array[:,end])) .+ offset]); offset += length(element_array[:,end])
    facesets["left"]   = Set{FaceIndex}(boundary[(1:length(element_array[1,:]))   .+ offset]); offset += length(element_array[1,:])

    return Mesh(elements, nodes, facesets=facesets)
end

# QuadraticQuadrilateral
function generate_mesh(::Type{QuadraticQuadrilateralElement}, nel::NTuple{2,Int}, LL::Vec{2,T}, LR::Vec{2,T}, UR::Vec{2,T}, UL::Vec{2,T}) where {T}
    nel_x = nel[1]; nel_y = nel[2]; nel_tot = nel_x*nel_y
    n_nodes_x = 2*nel_x + 1; n_nodes_y = 2*nel_y + 1
    n_nodes = n_nodes_x * n_nodes_y

    # Generate nodes
    nodes = Node{2,T}[]
    _generate_2d_nodes!(nodes, n_nodes_x, n_nodes_y, LL, LR, UR, UL)

    # Generate elements
    node_array = reshape(collect(1:n_nodes), (n_nodes_x, n_nodes_y))
    elements = QuadraticQuadrilateralElement[]
    for j in 1:nel_y, i in 1:nel_x
        push!(elements, QuadraticQuadrilateralElement((node_array[2*i-1,2*j-1],node_array[2*i+1,2*j-1],node_array[2*i+1,2*j+1],node_array[2*i-1,2*j+1],
                                             node_array[2*i,2*j-1],node_array[2*i+1,2*j],node_array[2*i,2*j+1],node_array[2*i-1,2*j],
                                             node_array[2*i,2*j])))
    end

    # Element faces
    element_array = reshape(collect(1:nel_tot),(nel_x, nel_y))
    boundary = FaceIndex[[FaceIndex(cl, 1) for cl in element_array[:,1]];
                              [FaceIndex(cl, 2) for cl in element_array[end,:]];
                              [FaceIndex(cl, 3) for cl in element_array[:,end]];
                              [FaceIndex(cl, 4) for cl in element_array[1,:]]]

    # Element face sets
    offset = 0
    facesets = Dict{String, Set{FaceIndex}}()
    facesets["bottom"] = Set{FaceIndex}(boundary[(1:length(element_array[:,1]))   .+ offset]); offset += length(element_array[:,1])
    facesets["right"]  = Set{FaceIndex}(boundary[(1:length(element_array[end,:])) .+ offset]); offset += length(element_array[end,:])
    facesets["top"]    = Set{FaceIndex}(boundary[(1:length(element_array[:,end])) .+ offset]); offset += length(element_array[:,end])
    facesets["left"]   = Set{FaceIndex}(boundary[(1:length(element_array[1,:]))   .+ offset]); offset += length(element_array[1,:])

    return Mesh(elements, nodes, facesets=facesets, boundary_matrix=boundary_matrix)
end

# Hexahedron
function generate_mesh(::Type{HexahedronElement}, nel::NTuple{3,Int}, left::Vec{3,T}=Vec{3}((-1.0,-1.0,-1.0)), right::Vec{3,T}=Vec{3}((1.0,1.0,1.0))) where {T}
    nel_x = nel[1]; nel_y = nel[2]; nel_z = nel[3]; nel_tot = nel_x*nel_y*nel_z
    n_nodes_x = nel_x + 1; n_nodes_y = nel_y + 1; n_nodes_z = nel_z + 1
    n_nodes = n_nodes_x * n_nodes_y * n_nodes_z

    # Generate nodes
    coords_x = range(left[1], stop=right[1], length=n_nodes_x)
    coords_y = range(left[2], stop=right[2], length=n_nodes_y)
    coords_z = range(left[3], stop=right[3], length=n_nodes_z)
    nodes = Node{3,T}[]
    for k in 1:n_nodes_z, j in 1:n_nodes_y, i in 1:n_nodes_x
        push!(nodes, Node((coords_x[i], coords_y[j], coords_z[k])))
    end

    # Generate elements
    node_array = reshape(collect(1:n_nodes), (n_nodes_x, n_nodes_y, n_nodes_z))
    elements = HexahedronElement[]
    for k in 1:nel_z, j in 1:nel_y, i in 1:nel_x
        push!(elements, HexahedronElement((node_array[i,j,k], node_array[i+1,j,k], node_array[i+1,j+1,k], node_array[i,j+1,k],
                                 node_array[i,j,k+1], node_array[i+1,j,k+1], node_array[i+1,j+1,k+1], node_array[i,j+1,k+1])))
    end

    # Element faces
    element_array = reshape(collect(1:nel_tot),(nel_x, nel_y, nel_z))
    boundary = FaceIndex[[FaceIndex(cl, 1) for cl in element_array[:,:,1][:]];
                              [FaceIndex(cl, 2) for cl in element_array[:,1,:][:]];
                              [FaceIndex(cl, 3) for cl in element_array[end,:,:][:]];
                              [FaceIndex(cl, 4) for cl in element_array[:,end,:][:]];
                              [FaceIndex(cl, 5) for cl in element_array[1,:,:][:]];
                              [FaceIndex(cl, 6) for cl in element_array[:,:,end][:]]]

    # Element face sets
    offset = 0
    facesets = Dict{String,Set{FaceIndex}}()
    facesets["bottom"] = Set{FaceIndex}(boundary[(1:length(element_array[:,:,1][:]))   .+ offset]); offset += length(element_array[:,:,1][:])
    facesets["front"]  = Set{FaceIndex}(boundary[(1:length(element_array[:,1,:][:]))   .+ offset]); offset += length(element_array[:,1,:][:])
    facesets["right"]  = Set{FaceIndex}(boundary[(1:length(element_array[end,:,:][:])) .+ offset]); offset += length(element_array[end,:,:][:])
    facesets["back"]   = Set{FaceIndex}(boundary[(1:length(element_array[:,end,:][:])) .+ offset]); offset += length(element_array[:,end,:][:])
    facesets["left"]   = Set{FaceIndex}(boundary[(1:length(element_array[1,:,:][:]))   .+ offset]); offset += length(element_array[1,:,:][:])
    facesets["top"]    = Set{FaceIndex}(boundary[(1:length(element_array[:,:,end][:])) .+ offset]); offset += length(element_array[:,:,end][:])

    return Mesh(elements, nodes, facesets=facesets)
end 

function Ferrite.generate_mesh(::Type{Element{3,RefCube,20}}, nel::NTuple{3,Int}, left::Vec{3,T}=Vec{3}((-1.0,-1.0,-1.0)), right::Vec{3,T}=Vec{3}((1.0,1.0,1.0))) where {T}
    nel_x = nel[1]; nel_y = nel[2]; nel_z = nel[3]; nel_tot = nel_x*nel_y*nel_z
    nnode_x = 2nel_x + 1; nnode_y = 2nel_y + 1; nnode_z = 2nel_z + 1 #Note: not the actually number of nodes in x/y/z, just a temporary variables

    # Generate nodes
    coords_x = range(left[1], stop=right[1], length=nnode_x)
    coords_y = range(left[2], stop=right[2], length=nnode_y)
    coords_z = range(left[3], stop=right[3], length=nnode_z)
    nodes = Node{3,T}[]

    node_array = fill(0, (nnode_x,nnode_y,nnode_z))
    nodeid = 0
    for k in 1:nnode_z, j in 1:nnode_y, i in 1:nnode_x
        (iseven(i) && iseven(j)) && continue
        (iseven(i) && iseven(k)) && continue
        (iseven(k) && iseven(j)) && continue
        push!(nodes, Node((coords_x[i], coords_y[j], coords_z[k])))
        nodeid += 1
        node_array[i,j,k] = nodeid
    end


    # Generate elements
    elements = Element{3,RefCube,20}[]
    for k in 1:2:2nel_z, j in 1:2:2nel_y, i in 1:2:2nel_x     
        push!(elements, Element{3,RefCube,20}((
                node_array[i,j,k], node_array[i+2,j,k], node_array[i+2,j+2,k], node_array[i,j+2,k], # vertices bot 
                node_array[i,j,k+2], node_array[i+2,j,k+2], node_array[i+2,j+2,k+2], node_array[i,j+2,k+2], # vertices top
                node_array[i+1,j,k], node_array[i+2,j+1,k], node_array[i+1,j+2,k], node_array[i,j+1,k], # edges horizontal bottom
                node_array[i+1,j,k+2], node_array[i+2,j+1,k+2], node_array[i+1,j+2,k+2], node_array[i,j+1,k+2], # edges horizontal top
                node_array[i,j,k+1], node_array[i+2,j,k+1], node_array[i+2,j+2,k+1], node_array[i,j+2,k+1] ))  # edges vertical
            )
    end

    # Element faces
    element_array = reshape(collect(1:nel_tot),(nel_x, nel_y, nel_z))
    boundary = FaceIndex[[FaceIndex(cl, 1) for cl in element_array[:,:,1][:]];
                              [FaceIndex(cl, 2) for cl in element_array[:,1,:][:]];
                              [FaceIndex(cl, 3) for cl in element_array[end,:,:][:]];
                              [FaceIndex(cl, 4) for cl in element_array[:,end,:][:]];
                              [FaceIndex(cl, 5) for cl in element_array[1,:,:][:]];
                              [FaceIndex(cl, 6) for cl in element_array[:,:,end][:]]]

    # Element face sets
    offset = 0
    facesets = Dict{String,Set{FaceIndex}}()
    facesets["bottom"] = Set{FaceIndex}(boundary[(1:length(element_array[:,:,1][:]))   .+ offset]); offset += length(element_array[:,:,1][:])
    facesets["front"]  = Set{FaceIndex}(boundary[(1:length(element_array[:,1,:][:]))   .+ offset]); offset += length(element_array[:,1,:][:])
    facesets["right"]  = Set{FaceIndex}(boundary[(1:length(element_array[end,:,:][:])) .+ offset]); offset += length(element_array[end,:,:][:])
    facesets["back"]   = Set{FaceIndex}(boundary[(1:length(element_array[:,end,:][:])) .+ offset]); offset += length(element_array[:,end,:][:])
    facesets["left"]   = Set{FaceIndex}(boundary[(1:length(element_array[1,:,:][:]))   .+ offset]); offset += length(element_array[1,:,:][:])
    facesets["top"]    = Set{FaceIndex}(boundary[(1:length(element_array[:,:,end][:])) .+ offset]); offset += length(element_array[:,:,end][:])

    return Mesh(elements, nodes, facesets=facesets)   
end

# Triangle
function generate_mesh(::Type{TriangleElement}, nel::NTuple{2,Int}, LL::Vec{2,T}, LR::Vec{2,T}, UR::Vec{2,T}, UL::Vec{2,T}) where {T}
    nel_x = nel[1]; nel_y = nel[2]; nel_tot = 2*nel_x*nel_y
    n_nodes_x = nel_x + 1; n_nodes_y = nel_y + 1
    n_nodes = n_nodes_x * n_nodes_y

    # Generate nodes
    nodes = Node{2,T}[]
    _generate_2d_nodes!(nodes, n_nodes_x, n_nodes_y, LL, LR, UR, UL)

    # Generate elements
    node_array = reshape(collect(1:n_nodes), (n_nodes_x, n_nodes_y))
    elements = TriangleElement[]
    for j in 1:nel_y, i in 1:nel_x
        push!(elements, TriangleElement((node_array[i,j], node_array[i+1,j], node_array[i,j+1]))) # ◺
        push!(elements, TriangleElement((node_array[i+1,j], node_array[i+1,j+1], node_array[i,j+1]))) # ◹
    end

    # Element faces
    element_array = reshape(collect(1:nel_tot),(2, nel_x, nel_y))
    boundary = FaceIndex[[FaceIndex(cl, 1) for cl in element_array[1,:,1]];
                               [FaceIndex(cl, 1) for cl in element_array[2,end,:]];
                               [FaceIndex(cl, 2) for cl in element_array[2,:,end]];
                               [FaceIndex(cl, 3) for cl in element_array[1,1,:]]]

    # Element face sets
    offset = 0
    facesets = Dict{String,Set{FaceIndex}}()
    facesets["bottom"] = Set{FaceIndex}(boundary[(1:length(element_array[1,:,1]))   .+ offset]); offset += length(element_array[1,:,1])
    facesets["right"]  = Set{FaceIndex}(boundary[(1:length(element_array[2,end,:])) .+ offset]); offset += length(element_array[2,end,:])
    facesets["top"]    = Set{FaceIndex}(boundary[(1:length(element_array[2,:,end])) .+ offset]); offset += length(element_array[2,:,end])
    facesets["left"]   = Set{FaceIndex}(boundary[(1:length(element_array[1,1,:]))   .+ offset]); offset += length(element_array[1,1,:])

    return Mesh(elements, nodes, facesets=facesets)
end

# QuadraticTriangle
function generate_mesh(::Type{QuadraticTriangleElement}, nel::NTuple{2,Int}, LL::Vec{2,T}, LR::Vec{2,T}, UR::Vec{2,T}, UL::Vec{2,T}) where {T}
    nel_x = nel[1]; nel_y = nel[2]; nel_tot = 2*nel_x*nel_y
    n_nodes_x = 2*nel_x + 1; n_nodes_y = 2*nel_y + 1
    n_nodes = n_nodes_x * n_nodes_y

    # Generate nodes
    nodes = Node{2,T}[]
    _generate_2d_nodes!(nodes, n_nodes_x, n_nodes_y, LL, LR, UR, UL)

    # Generate elements
    node_array = reshape(collect(1:n_nodes), (n_nodes_x, n_nodes_y))
    elements = QuadraticTriangleElement[]
    for j in 1:nel_y, i in 1:nel_x
        push!(elements, QuadraticTriangleElement((node_array[2*i-1,2*j-1], node_array[2*i+1,2*j-1], node_array[2*i-1,2*j+1],
                                        node_array[2*i,2*j-1], node_array[2*i,2*j], node_array[2*i-1,2*j]))) # ◺
        push!(elements, QuadraticTriangleElement((node_array[2*i+1,2*j-1], node_array[2*i+1,2*j+1], node_array[2*i-1,2*j+1],
                                        node_array[2*i+1,2*j], node_array[2*i,2*j+1], node_array[2*i,2*j]))) # ◹
    end

    # Element faces
    element_array = reshape(collect(1:nel_tot),(2, nel_x, nel_y))
    boundary = FaceIndex[[FaceIndex(cl, 1) for cl in element_array[1,:,1]];
                              [FaceIndex(cl, 1) for cl in element_array[2,end,:]];
                              [FaceIndex(cl, 2) for cl in element_array[2,:,end]];
                              [FaceIndex(cl, 3) for cl in element_array[1,1,:]]]

    # Element face sets
    offset = 0
    facesets = Dict{String,Set{FaceIndex}}()
    facesets["bottom"] = Set{FaceIndex}(boundary[(1:length(element_array[1,:,1]))   .+ offset]); offset += length(element_array[1,:,1])
    facesets["right"]  = Set{FaceIndex}(boundary[(1:length(element_array[2,end,:])) .+ offset]); offset += length(element_array[2,end,:])
    facesets["top"]    = Set{FaceIndex}(boundary[(1:length(element_array[2,:,end])) .+ offset]); offset += length(element_array[2,:,end])
    facesets["left"]   = Set{FaceIndex}(boundary[(1:length(element_array[1,1,:]))   .+ offset]); offset += length(element_array[1,1,:])

    return Mesh(elements, nodes, facesets=facesets, boundary_matrix)
end

# Tetrahedron
function generate_mesh(::Type{TetrahedronElement}, elements_per_dim::NTuple{3,Int}, left::Vec{3,T}=Vec{3}((-1.0,-1.0,-1.0)), right::Vec{3,T}=Vec{3}((1.0,1.0,1.0))) where {T}
    nodes_per_dim = elements_per_dim .+ 1

    elements_per_cube = 6
    total_nodes = prod(nodes_per_dim)
    total_elements = elements_per_cube * prod(elements_per_dim)

    n_nodes_x, n_nodes_y, n_nodes_z = nodes_per_dim
    n_elements_x, n_elements_y, n_elements_z = elements_per_dim

    # Generate nodes
    coords_x = range(left[1], stop=right[1], length=n_nodes_x)
    coords_y = range(left[2], stop=right[2], length=n_nodes_y)
    coords_z = range(left[3], stop=right[3], length=n_nodes_z)
    numbering = reshape(1:total_nodes, nodes_per_dim)

    # Pre-allocate the nodes & elements
    nodes = Vector{Node{3,T}}(undef, total_nodes)
    elements = Vector{TetrahedronElement}(undef, total_elements)

    # Generate nodes
    node_idx = 1
    @inbounds for k in 1:n_nodes_z, j in 1:n_nodes_y, i in 1:n_nodes_x
        nodes[node_idx] = Node((coords_x[i], coords_y[j], coords_z[k]))
        node_idx += 1
    end

    # Generate elements, case 1 from: http://www.baumanneduard.ch/Splitting%20a%20cube%20in%20tetrahedras2.htm
    # cube = (1, 2, 3, 4, 5, 6, 7, 8)
    # left = (1, 4, 5, 8), right = (2, 3, 6, 7)
    # front = (1, 2, 5, 6), back = (3, 4, 7, 8)
    # bottom = (1, 2, 3, 4), top = (5, 6, 7, 8)
    element_idx = 0
    @inbounds for k in 1:n_elements_z, j in 1:n_elements_y, i in 1:n_elements_x
        element = (
            numbering[i  , j  , k],
            numbering[i+1, j  , k],
            numbering[i+1, j+1, k],
            numbering[i  , j+1, k],
            numbering[i  , j  , k+1],
            numbering[i+1, j  , k+1],
            numbering[i+1, j+1, k+1],
            numbering[i  , j+1, k+1]
        )

        elements[element_idx + 1] = TetrahedronElement((element[1], element[2], element[4], element[8]))
        elements[element_idx + 2] = TetrahedronElement((element[1], element[5], element[2], element[8]))
        elements[element_idx + 3] = TetrahedronElement((element[2], element[3], element[4], element[8]))
        elements[element_idx + 4] = TetrahedronElement((element[2], element[7], element[3], element[8]))
        elements[element_idx + 5] = TetrahedronElement((element[2], element[5], element[6], element[8]))
        elements[element_idx + 6] = TetrahedronElement((element[2], element[6], element[7], element[8]))

        element_idx += elements_per_cube
    end

    # Order the elements as c_nxyz[n, x, y, z] such that we can look up boundary elements
    c_nxyz = reshape(1:total_elements, (elements_per_cube, elements_per_dim...))

    @views le = [map(x -> FaceIndex(x,4), c_nxyz[1, 1, :, :][:])   ; map(x -> FaceIndex(x,2), c_nxyz[2, 1, :, :][:])]
    @views ri = [map(x -> FaceIndex(x,1), c_nxyz[4, end, :, :][:]) ; map(x -> FaceIndex(x,1), c_nxyz[6, end, :, :][:])]
    @views fr = [map(x -> FaceIndex(x,1), c_nxyz[2, :, 1, :][:])   ; map(x -> FaceIndex(x,1), c_nxyz[5, :, 1, :][:])]
    @views ba = [map(x -> FaceIndex(x,3), c_nxyz[3, :, end, :][:]) ; map(x -> FaceIndex(x,3), c_nxyz[4, :, end, :][:])]
    @views bo = [map(x -> FaceIndex(x,1), c_nxyz[1, :, :, 1][:])   ; map(x -> FaceIndex(x,1), c_nxyz[3, :, :, 1][:])]
    @views to = [map(x -> FaceIndex(x,3), c_nxyz[5, :, :, end][:]) ; map(x -> FaceIndex(x,3), c_nxyz[6, :, :, end][:])]

    facesets = Dict(
        "left" => Set(le),
        "right" => Set(ri),
        "front" => Set(fr),
        "back" => Set(ba),
        "bottom" => Set(bo),
        "top" => Set(to),
    )

    return Mesh(elements, nodes, facesets=facesets)
end
