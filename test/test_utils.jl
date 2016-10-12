# Some utility functions for testing JuAFEM.jl

#####################################
# Volume for the reference elements #
#####################################
reference_volume{dim}(::FunctionSpace{dim, RefCube}) = 2^dim
reference_volume{dim}(::FunctionSpace{dim, RefTetrahedron}) = 1 / factorial(dim)
# For boundaries
reference_volume(fs::FunctionSpace, ::Int) = reference_volume(JuAFEM.functionspace_lower_dim(fs))
reference_volume(fs::FunctionSpace{2, RefTetrahedron}, boundary::Int) = boundary == 1 ? sqrt(2) : 1.0
reference_volume(fs::FunctionSpace{3, RefTetrahedron}, boundary::Int) = boundary == 3 ? sqrt(2 * 1.5) / 2.0 : 0.5

##########################################
# Coordinates for the reference elements #
##########################################
function reference_coordinates(::Lagrange{1, RefCube, 1})
    return [Vec{1, Float64}((-1.0,)),
            Vec{1, Float64}(( 1.0,))]
end

function reference_coordinates(::Lagrange{1, RefCube, 2})
    return [Vec{1, Float64}((-1.0,)),
            Vec{1, Float64}(( 1.0,)),
            Vec{1, Float64}(( 0.0,))]
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

function reference_coordinates(::Lagrange{3, RefTetrahedron, 1})
    return [Vec{3, Float64}((1.0, 0.0, 0.0)),
            Vec{3, Float64}((0.0, 1.0, 0.0)),
            Vec{3, Float64}((0.0, 0.0, 1.0)),
            Vec{3, Float64}((0.0, 0.0, 0.0))]
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
        u = Vec{3}((1.0, 2.0, 3.0)); u *= norm(u)
        ux = Tensor{2,3}((0.0, u[3], -u[2], -u[3], 0.0, u[1], u[2], -u[1], 0.0))
        R = cos(θ)*one(Tensor{2,3}) + sin(θ)*ux + (1-cos(θ))*u⊗u
        return R
    end
end

function valid_coordinates{dim, shape, order}(fs::FunctionSpace{dim, shape, order})
    x = reference_coordinates(fs)
    R = rotmat(dim)
    return [2.0 * (R ⋅ x[i]) for i in 1:length(x)]
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

function calculate_volume{T}(::Lagrange{3, RefTetrahedron, 1}, x::Vector{Vec{3, T}})
    vol = norm((x[1] - x[4]) ⋅ ((x[2] - x[4]) × (x[3] - x[4]))) / 6.0
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

# For boundaries
function calculate_volume{order, T}(::Lagrange{0, RefCube, order}, ::Vector{Vec{1, T}})
    return one(T)
end

####################################
# For testing get_boundarynumber() #
####################################
# Last set of boundary nodes throws error
function topology_test_nodes(::Lagrange{1, RefCube, 1})
    cell_nodes = [3,4]
    boundary_nodes = [[3,], [4,], [1337,]]
    return boundary_nodes, cell_nodes
end
function topology_test_nodes(::Lagrange{1, RefCube, 2})
    cell_nodes = [3,4,8]
    boundary_nodes = [[3,], [4,], [8,]]
    return boundary_nodes, cell_nodes
end
function topology_test_nodes(::Lagrange{2, RefCube, 1})
    cell_nodes = [3,4,8,1]
    boundary_nodes = [[3,4], [4,8], [8,1], [1,3], [3,1337]]
    return boundary_nodes, cell_nodes
end
function topology_test_nodes(::Lagrange{2, RefCube, 2})
    cell_nodes = [3,4,8,1,2,5,6,7,9]
    boundary_nodes = [[3,4,2], [4,8,5], [8,1,6], [1,3,7], [3,4,1337]]
    return boundary_nodes, cell_nodes
end
function topology_test_nodes(::Lagrange{2, RefTetrahedron, 1})
    cell_nodes = [3,4,8]
    boundary_nodes = [[3,4], [4,8], [8,3], [3,1337]]
    return boundary_nodes, cell_nodes
end
function topology_test_nodes(::Lagrange{2, RefTetrahedron, 2})
    cell_nodes = [3,4,8,1,2,5]
    boundary_nodes = [[3,4,1], [4,8,2], [8,3,5], [3,4,1337]]
    return boundary_nodes, cell_nodes
end
function topology_test_nodes(::Lagrange{3, RefCube, 1})
    cell_nodes = [3,4,8,1,2,5,6,7]
    boundary_nodes = [[3,4,8,1], [3,4,5,2], [4,8,6,5], [8,1,7,6], [1,3,2,7], [2,5,6,7], [3,4,8,1337]]
    return boundary_nodes, cell_nodes
end
function topology_test_nodes(::Serendipity{2, RefCube, 2})
    cell_nodes = [3,4,8,1,2,5,6,7]
    boundary_nodes = [[3,4,2], [4,8,5], [8,1,6], [1,3,7], [3,4,1337]]
    return boundary_nodes, cell_nodes
end
function topology_test_nodes(::Lagrange{3, RefTetrahedron, 1})
    cell_nodes = [3,4,8,1]
    boundary_nodes = [[1,4,3], [1,3,8], [3,4,8], [1,8,4], [1,4,1337]]
    return boundary_nodes, cell_nodes
end
