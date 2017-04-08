# Some utility functions for testing JuAFEM.jl

"""
The face number for a cell, typically used to get the face number which is needed
to `reinit!` a `FaceValues` object for  face integration

    getfacenumber(face_nodes, cell_nodes, ip::Interpolation)

** Arguments **

* `face_nodes`: the node numbers of the nodes on the face of the cell
* `cell_nodes`: the node numbers of the cell
* `ip`: the `Interpolation` for the cell

** Results **

* `::Int`: the corresponding face
"""
function getfacenumber(face_nodes::Vector{Int}, cell_nodes::Vector{Int}, ip::Interpolation)
    @assert length(face_nodes) == getnfacenodes(ip)
    @assert length(cell_nodes) == getnbasefunctions(ip)

    tmp = zeros(face_nodes)
    for i in 1:length(face_nodes)
        tmp[i] = findfirst(j -> j == face_nodes[i], cell_nodes)
    end

    if 0 in tmp
        throw(ArgumentError("at least one face node: $face_nodes not in cell nodes: $cell_nodes"))
    end
    sort!(tmp)
    face_nodes_sorted = ntuple(i -> tmp[i], Val{getnfacenodes(ip)})
    for (i, face) in enumerate(getfacelist(ip))
        face_nodes_sorted == face && return i
    end

    throw(ArgumentError("invalid node numbers for face"))
end


getnfaces{dim}(::Interpolation{dim, RefCube}) = 2*dim
getnfaces(::Interpolation{2, RefTetrahedron}) = 3
getnfaces(::Interpolation{3, RefTetrahedron}) = 4

getfacelist(i::Interpolation) = getfacelist(typeof(i))

getnfacenodes(::Lagrange{1, RefCube, 1}) = 1
getfacelist(::Type{Lagrange{1, RefCube, 1}}) = ((1,),(2,))

getnfacenodes(::Lagrange{1, RefCube, 2}) = 1
getfacelist(::Type{Lagrange{1, RefCube, 2}}) = ((1,),(2,))

getnfacenodes(::Lagrange{2, RefCube, 1}) = 2
getfacelist(::Type{Lagrange{2, RefCube, 1}}) = ((1,2),(2,3),(3,4),(1,4))

getnfacenodes(::Lagrange{2, RefCube, 2}) = 3
getfacelist(::Type{Lagrange{2, RefCube, 2}}) = ((1,2,5),(2,3,6),(3,4,7),(1,4,8))

getlowerdim{order}(::Lagrange{2, RefTetrahedron, order}) = Lagrange{1, RefCube, order}()
getnfacenodes(::Lagrange{2, RefTetrahedron, 1}) = 2
getfacelist(::Type{Lagrange{2, RefTetrahedron, 1}}) = ((1,2),(2,3),(1,3))

getnfacenodes(::Lagrange{2, RefTetrahedron, 2}) = 3
getfacelist(::Type{Lagrange{2, RefTetrahedron, 2}}) = ((1,2,4),(2,3,5),(1,3,6))

getnfacenodes(::Lagrange{3, RefTetrahedron, 1}) = 3
getfacelist(::Type{Lagrange{3, RefTetrahedron, 1}}) = ((1,2,3),(1,2,4),(2,3,4),(1,3,4))

getnfacenodes(::Lagrange{3, RefTetrahedron, 2}) = 6
getfacelist(::Lagrange{3, RefTetrahedron, 2}) = ((1,2,3,5,6,7),(1,2,4,5,8,9),(2,3,4,6,9,10),(1,3,4,7,8,10))

getnfacenodes(::Lagrange{3, RefCube, 1}) = 4
getfacelist(::Type{Lagrange{3, RefCube, 1}}) = ((1,2,3,4),(1,2,5,6),(2,3,6,7),(3,4,7,8),(1,4,5,8),(5,6,7,8))

getlowerdim{dim,shape,order}(::Lagrange{dim,shape,order}) = Lagrange{dim-1,shape,order}()
getlowerorder{dim,shape,order}(::Lagrange{dim,shape,order}) = Lagrange{dim,shape,order-1}()

getlowerdim(::Serendipity{2, RefCube, 2}) = Lagrange{1, RefCube, 2}()
getlowerorder(::Serendipity{2, RefCube, 2}) = Lagrange{2, RefCube, 1}()
getnfacenodes(::Serendipity{2, RefCube, 2}) = 3
getfacelist(::Type{Serendipity{2, RefCube, 2}}) = ((1,2,5),(2,3,6),(3,4,7),(1,4,8))


#####################################
# Volume for the reference elements #
#####################################
reference_volume{dim}(::Interpolation{dim, RefCube}) = 2^dim
reference_volume{dim}(::Interpolation{dim, RefTetrahedron}) = 1 / factorial(dim)
# For faces
reference_volume(fs::Interpolation, ::Int) = reference_volume(getlowerdim(fs))
reference_volume(fs::Interpolation{2, RefTetrahedron}, face::Int) = face == 1 ? sqrt(2) : 1.0
reference_volume(fs::Interpolation{3, RefTetrahedron}, face::Int) = face == 3 ? sqrt(2 * 1.5) / 2.0 : 0.5

######################################################
# Coordinates and normals for the reference elements #
######################################################

# Lagrange{1, RefCube}
function reference_normals(::Lagrange{1, RefCube})
    return [Vec{1, Float64}((-1.0,)),
            Vec{1, Float64}(( 1.0,))]
end

function reference_coordinates(::Lagrange{1, RefCube, 1})
    return [Vec{1, Float64}((-1.0,)),
            Vec{1, Float64}(( 1.0,))]
end

function reference_coordinates(::Lagrange{1, RefCube, 2})
    return [Vec{1, Float64}((-1.0,)),
            Vec{1, Float64}(( 1.0,)),
            Vec{1, Float64}(( 0.0,))]
end

# Lagrange{2, RefCube}
function reference_normals(::Lagrange{2, RefCube})
    return [Vec{2, Float64}(( 0.0, -1.0)),
            Vec{2, Float64}(( 1.0,  0.0)),
            Vec{2, Float64}(( 0.0,  1.0,)),
            Vec{2, Float64}((-1.0,  0.0,))]
end

function reference_coordinates(::Lagrange{2, RefCube, 1})
    return [Vec{2, Float64}((-1.0, -1.0)),
            Vec{2, Float64}(( 1.0, -1.0)),
            Vec{2, Float64}(( 1.0,  1.0,)),
            Vec{2, Float64}((-1.0,  1.0,))]
end

function reference_coordinates(::Lagrange{2, RefCube, 2})
    return [Vec{2, Float64}((-1.0, -1.0)),
            Vec{2, Float64}(( 1.0, -1.0)),
            Vec{2, Float64}(( 1.0,  1.0)),
            Vec{2, Float64}((-1.0,  1.0)),
            Vec{2, Float64}(( 0.0, -1.0)),
            Vec{2, Float64}(( 1.0,  0.0)),
            Vec{2, Float64}(( 0.0,  1.0)),
            Vec{2, Float64}((-1.0,  0.0)),
            Vec{2, Float64}(( 0.0,  0.0))]
end


# Lagrange{2, RefTetrahedron}
function reference_normals(::Lagrange{2, RefTetrahedron})
    return [Vec{2, Float64}((1/√2, 1/√2)),
            Vec{2, Float64}((-1.0, 0.0)),
            Vec{2, Float64}((0.0, -1.0))]
end

function reference_coordinates(::Lagrange{2, RefTetrahedron, 1})
    return [Vec{2, Float64}((1.0, 0.0)),
            Vec{2, Float64}((0.0, 1.0)),
            Vec{2, Float64}((0.0, 0.0))]
end

function reference_coordinates(::Lagrange{2, RefTetrahedron, 2})
    return [Vec{2, Float64}((1.0, 0.0)),
            Vec{2, Float64}((0.0, 1.0)),
            Vec{2, Float64}((0.0, 0.0)),
            Vec{2, Float64}((0.5, 0.5)),
            Vec{2, Float64}((0.0, 0.5)),
            Vec{2, Float64}((0.5, 0.0))]
end

# Lagrange{3, RefTetrahedron}
function reference_normals(::Lagrange{3, RefTetrahedron})
    return [Vec{3, Float64}((0.0, 0.0, -1.0)),
            Vec{3, Float64}((0.0 ,-1.0, 0.0)),
            Vec{3, Float64}((1/√3, 1/√3, 1/√3)),
            Vec{3, Float64}((-1.0, 0.0, 0.0))]
end

function reference_coordinates(::Lagrange{3, RefTetrahedron, 1})
    return [Vec{3, Float64}((0.0, 0.0, 0.0)),
            Vec{3, Float64}((1.0, 0.0, 0.0)),
            Vec{3, Float64}((0.0, 1.0, 0.0)),
            Vec{3, Float64}((0.0, 0.0, 1.0))]
end

function reference_coordinates(::Lagrange{3, RefTetrahedron, 2})
    return [Vec{3, Float64}((0.0, 0.0, 0.0)),
            Vec{3, Float64}((1.0, 0.0, 0.0)),
            Vec{3, Float64}((0.0, 1.0, 0.0)),
            Vec{3, Float64}((0.0, 0.0, 1.0)),
            Vec{3, Float64}((0.5, 0.0, 0.0)),
            Vec{3, Float64}((0.5, 0.5, 0.0)),
            Vec{3, Float64}((0.0, 0.5, 0.0)),
            Vec{3, Float64}((0.0, 0.0, 0.5)),
            Vec{3, Float64}((0.5, 0.0, 0.5)),
            Vec{3, Float64}((0.0, 0.5, 0.5))]
end

# Lagrange{3, Cube}
function reference_normals(::Lagrange{3, RefCube, 1})
    return [Vec{3, Float64}(( 0.0,  0.0, -1.0)),
            Vec{3, Float64}(( 0.0, -1.0,  0.0)),
            Vec{3, Float64}(( 1.0,  0.0,  0.0)),
            Vec{3, Float64}(( 0.0,  1.0,  0.0)),
            Vec{3, Float64}((-1.0,  0.0,  0.0)),
            Vec{3, Float64}(( 0.0,  0.0,  1.0))]
end

function reference_coordinates(::Lagrange{3, RefCube, 1})
    return [Vec{3, Float64}((-1.0, -1.0, -1.0)),
            Vec{3, Float64}(( 1.0, -1.0, -1.0)),
            Vec{3, Float64}(( 1.0,  1.0, -1.0)),
            Vec{3, Float64}((-1.0,  1.0, -1.0)),
            Vec{3, Float64}((-1.0, -1.0,  1.0)),
            Vec{3, Float64}(( 1.0, -1.0,  1.0)),
            Vec{3, Float64}(( 1.0,  1.0,  1.0)),
            Vec{3, Float64}((-1.0,  1.0,  1.0))]
end

# Serendipity{2, RefCube}
reference_normals(::Serendipity{2, RefCube, 2}) = reference_normals(Lagrange{2, RefCube,1}())

function reference_coordinates(::Serendipity{2, RefCube, 2})
    return [Vec{2, Float64}((-1.0, -1.0)),
            Vec{2, Float64}(( 1.0, -1.0)),
            Vec{2, Float64}(( 1.0,  1.0)),
            Vec{2, Float64}((-1.0,  1.0)),
            Vec{2, Float64}(( 0.0, -1.0)),
            Vec{2, Float64}(( 1.0,  0.0)),
            Vec{2, Float64}(( 0.0,  1.0)),
            Vec{2, Float64}((-1.0,  0.0))]
end

##################################
# Valid coordinates by expanding #
# and rotating reference shape   #
##################################
function rotmat(dim, θ=π/6)
    if dim == 1
        R = Tensor{2,1}((cos(θ),))
        return R
    elseif dim == 2
        R = Tensor{2,2}((cos(θ), sin(θ), -sin(θ), cos(θ)))
        return R
    else
        u = Vec{3}((1.0, 2.0, 3.0)); u /= norm(u)
        ux = Tensor{2,3}((0.0, u[3], -u[2], -u[3], 0.0, u[1], u[2], -u[1], 0.0))
        R = cos(θ)*one(Tensor{2,3}) + sin(θ)*ux + (1-cos(θ))*u⊗u
        return R
    end
end

function valid_coordinates_and_normals{dim, shape, order}(fs::Interpolation{dim, shape, order})
    x = reference_coordinates(fs)
    n = reference_normals(fs)
    R = rotmat(dim)
    return [2.0 * (R ⋅ x[i]) for i in 1:length(x)] , [(R ⋅ n[i]) / norm((R ⋅ n[i])) for i in 1:length(n)]
end

#######################################
# Volume of cells (with planar edges) #
#######################################
function calculate_volume{T, dim}(::Lagrange{1, RefCube, 1}, x::Vector{Vec{dim, T}})
    vol = norm(x[2] - x[1])
    return vol
end

function calculate_volume{T, dim}(::Lagrange{1, RefCube, 2}, x::Vector{Vec{dim, T}})
    vol = norm(x[3] - x[1]) + norm(x[2]-x[3])
    return vol
end

function calculate_volume{T, dim}(::Lagrange{2, RefCube, 1}, x::Vector{Vec{dim, T}})
    vol = norm((x[4] - x[1]) × (x[2] - x[1])) * 0.5 +
          norm((x[4] - x[3]) × (x[2] - x[3])) * 0.5
    return vol
end

function calculate_volume{T, dim}(::Lagrange{2, RefCube, 2}, x::Vector{Vec{dim, T}})
    vol = norm((x[8] - x[1]) × (x[5] - x[1])) * 0.5 +
          norm((x[8] - x[9]) × (x[5] - x[9])) * 0.5 +
          norm((x[5] - x[2]) × (x[6] - x[2])) * 0.5 +
          norm((x[5] - x[9]) × (x[6] - x[9])) * 0.5 +
          norm((x[6] - x[3]) × (x[7] - x[3])) * 0.5 +
          norm((x[6] - x[9]) × (x[7] - x[9])) * 0.5 +
          norm((x[7] - x[4]) × (x[8] - x[4])) * 0.5 +
          norm((x[7] - x[9]) × (x[8] - x[9])) * 0.5
    return vol
end

function calculate_volume{T, dim}(::Lagrange{2, RefTetrahedron, 1}, x::Vector{Vec{dim, T}})
    vol = norm((x[1] - x[3]) × (x[2] - x[3])) * 0.5
    return vol
end

function calculate_volume{T, dim}(::Lagrange{2, RefTetrahedron, 2}, x::Vector{Vec{dim, T}})
    vol = norm((x[6] - x[3]) × (x[5] - x[3])) * 0.5 +
          norm((x[6] - x[4]) × (x[5] - x[4])) * 0.5 +
          norm((x[1] - x[6]) × (x[4] - x[6])) * 0.5 +
          norm((x[4] - x[5]) × (x[2] - x[5])) * 0.5
    return vol
end

function calculate_volume{T, order}(::Lagrange{3, RefTetrahedron, order}, x::Vector{Vec{3, T}})
    vol = norm((x[2] - x[1]) ⋅ ((x[3] - x[1]) × (x[4] - x[1]))) / 6.0
    return vol
end

function calculate_volume{T}(::Lagrange{3, RefCube, 1}, x::Vector{Vec{3, T}})
    vol = norm((x[1] - x[5]) ⋅ ((x[2] - x[5]) × (x[4] - x[5]))) / 6.0 +
          norm((x[2] - x[7]) ⋅ ((x[3] - x[7]) × (x[4] - x[7]))) / 6.0 +
          norm((x[2] - x[7]) ⋅ ((x[4] - x[7]) × (x[5] - x[7]))) / 6.0 +
          norm((x[2] - x[7]) ⋅ ((x[5] - x[7]) × (x[6] - x[7]))) / 6.0 +
          norm((x[4] - x[8]) ⋅ ((x[5] - x[8]) × (x[7] - x[8]))) / 6.0
    return vol
end

function calculate_volume{T}(::Serendipity{2, RefCube, 2}, x::Vector{Vec{2, T}})
    vol = norm((x[5] - x[1]) × (x[8] - x[1])) * 0.5 +
          norm((x[6] - x[2]) × (x[5] - x[2])) * 0.5 +
          norm((x[7] - x[3]) × (x[6] - x[3])) * 0.5 +
          norm((x[8] - x[4]) × (x[7] - x[4])) * 0.5 +
          norm((x[6] - x[5]) × (x[8] - x[5])) * 0.5 +
          norm((x[6] - x[7]) × (x[8] - x[7])) * 0.5
    return vol
end

# For faces
function calculate_volume{order, T}(::Lagrange{0, RefCube, order}, ::Vector{Vec{1, T}})
    return one(T)
end

####################################
# For testing getfacenumber() #
####################################
# Last set of face nodes throws error
function topology_test_nodes(::Lagrange{1, RefCube, 1})
    cell_nodes = [3,4]
    face_nodes = [[3,], [4,], [1337,]]
    return face_nodes, cell_nodes
end
function topology_test_nodes(::Lagrange{1, RefCube, 2})
    cell_nodes = [3,4,8]
    face_nodes = [[3,], [4,], [8,]]
    return face_nodes, cell_nodes
end
function topology_test_nodes(::Lagrange{2, RefCube, 1})
    cell_nodes = [3,4,8,1]
    face_nodes = [[3,4], [4,8], [8,1], [1,3], [3,1337]]
    return face_nodes, cell_nodes
end
function topology_test_nodes(::Lagrange{2, RefCube, 2})
    cell_nodes = [3,4,8,1,2,5,6,7,9]
    face_nodes = [[3,4,2], [4,8,5], [8,1,6], [1,3,7], [3,4,1337]]
    return face_nodes, cell_nodes
end
function topology_test_nodes(::Lagrange{2, RefTetrahedron, 1})
    cell_nodes = [3,4,8]
    face_nodes = [[3,4], [4,8], [8,3], [3,1337]]
    return face_nodes, cell_nodes
end
function topology_test_nodes(::Lagrange{2, RefTetrahedron, 2})
    cell_nodes = [3,4,8,1,2,5]
    face_nodes = [[3,4,1], [4,8,2], [8,3,5], [3,4,1337]]
    return face_nodes, cell_nodes
end
function topology_test_nodes(::Lagrange{3, RefCube, 1})
    cell_nodes = [3,4,8,1,2,5,6,7]
    face_nodes = [[3,4,8,1], [3,4,5,2], [4,8,6,5], [8,1,7,6], [1,3,2,7], [2,5,6,7], [3,4,8,1337]]
    return face_nodes, cell_nodes
end
function topology_test_nodes(::Serendipity{2, RefCube, 2})
    cell_nodes = [3,4,8,1,2,5,6,7]
    face_nodes = [[3,4,2], [4,8,5], [8,1,6], [1,3,7], [3,4,1337]]
    return face_nodes, cell_nodes
end
function topology_test_nodes(::Lagrange{3, RefTetrahedron, 1})
    cell_nodes = [3,4,8,1]
    face_nodes = [[3,4,8], [3,4,1], [4,8,1], [3,8,1], [1,4,1337]]
    return face_nodes, cell_nodes
end

function topology_test_nodes(::Lagrange{3, RefTetrahedron, 2})
    cell_nodes = [3,4,8,1,5,2,6,7,9,10]
    face_nodes = [[3,4,8,5,2,6], [3,4,1,5,9,7], [4,8,1,2,10,9], [3,8,1,6,10,7], [1,4,1337,2,3,5]]
    return face_nodes, cell_nodes
end
