# Some utility functions for testing Ferrite.jl

#####################################
# Volume for the reference elements #
#####################################
reference_volume(::Interpolation{dim, RefCube}) where {dim} = 2^dim
reference_volume(::Interpolation{dim, RefTetrahedron}) where {dim} = 1 / factorial(dim)
reference_volume(::Interpolation{  3, RefPrism}) = 1/2
# For faces
reference_volume(fs::Interpolation, ::Int) = reference_volume(Ferrite.getlowerdim(fs))
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

# Lagrange{2, RefCube}
function reference_normals(::Lagrange{2, RefCube})
    return [Vec{2, Float64}(( 0.0, -1.0)),
            Vec{2, Float64}(( 1.0,  0.0)),
            Vec{2, Float64}(( 0.0,  1.0,)),
            Vec{2, Float64}((-1.0,  0.0,))]
end

# Lagrange{2, RefTetrahedron}
function reference_normals(::Lagrange{2, RefTetrahedron})
    return [Vec{2, Float64}((1/√2, 1/√2)),
            Vec{2, Float64}((-1.0, 0.0)),
            Vec{2, Float64}((0.0, -1.0))]
end

# Lagrange{3, RefTetrahedron}
function reference_normals(::Lagrange{3, RefTetrahedron})
    return [Vec{3, Float64}((0.0, 0.0, -1.0)),
            Vec{3, Float64}((0.0 ,-1.0, 0.0)),
            Vec{3, Float64}((1/√3, 1/√3, 1/√3)),
            Vec{3, Float64}((-1.0, 0.0, 0.0))]
end

# Lagrange{3, Cube}
function reference_normals(::Lagrange{3, RefCube})
    return [Vec{3, Float64}(( 0.0,  0.0, -1.0)),
            Vec{3, Float64}(( 0.0, -1.0,  0.0)),
            Vec{3, Float64}(( 1.0,  0.0,  0.0)),
            Vec{3, Float64}(( 0.0,  1.0,  0.0)),
            Vec{3, Float64}((-1.0,  0.0,  0.0)),
            Vec{3, Float64}(( 0.0,  0.0,  1.0))]
end

# Lagrange{3, Wedge}
function reference_normals(::Lagrange{3, RefPrism})
    return [Vec{3, Float64}(( 0.0,  0.0, -1.0)),
            Vec{3, Float64}(( 0.0, -1.0,  0.0)),
            Vec{3, Float64}((-1.0,  0.0,  0.0)),
            Vec{3, Float64}((1/√2, 1/√2,  0.0)),
            Vec{3, Float64}(( 0.0,  0.0,  1.0))]
end

# Serendipity{2, RefCube}
reference_normals(::Serendipity{2, RefCube, 2}) = reference_normals(Lagrange{2, RefCube,1}())

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

function valid_coordinates_and_normals(fs::Interpolation{dim, shape, order}) where {dim, shape, order}
    x = Ferrite.reference_coordinates(fs)
    n = reference_normals(fs)
    R = rotmat(dim)
    return [2.0 * (R ⋅ x[i]) for i in 1:length(x)] , [(R ⋅ n[i]) / norm((R ⋅ n[i])) for i in 1:length(n)]
end

#######################################
# Volume of cells (with planar edges) #
#######################################
function calculate_volume(::Lagrange{1, RefCube, 1}, x::Vector{Vec{dim, T}}) where {T, dim}
    vol = norm(x[2] - x[1])
    return vol
end

function calculate_volume(::Lagrange{1, RefCube, 2}, x::Vector{Vec{dim, T}}) where {T, dim}
    vol = norm(x[3] - x[1]) + norm(x[2]-x[3])
    return vol
end

function calculate_volume(::Lagrange{2, RefCube, 1}, x::Vector{Vec{dim, T}}) where {T, dim}
    vol = norm((x[4] - x[1]) × (x[2] - x[1])) * 0.5 +
          norm((x[4] - x[3]) × (x[2] - x[3])) * 0.5
    return vol
end

function calculate_volume(::Lagrange{2, RefCube, 2}, x::Vector{Vec{dim, T}}) where {T, dim}
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

function calculate_volume(::Lagrange{2, RefTetrahedron, 1}, x::Vector{Vec{dim, T}}) where {T, dim}
    vol = norm((x[1] - x[3]) × (x[2] - x[3])) * 0.5
    return vol
end

function calculate_volume(::Lagrange{2, RefTetrahedron, 2}, x::Vector{Vec{dim, T}}) where {T, dim}
    vol = norm((x[6] - x[3]) × (x[5] - x[3])) * 0.5 +
          norm((x[6] - x[4]) × (x[5] - x[4])) * 0.5 +
          norm((x[1] - x[6]) × (x[4] - x[6])) * 0.5 +
          norm((x[4] - x[5]) × (x[2] - x[5])) * 0.5
    return vol
end

# TODO: Only correct for linear sides
function calculate_volume(::Lagrange{2, RefTetrahedron, O}, x::Vector{Vec{dim, T}}) where {T, dim, O}
    vol = norm((x[1] - x[3]) × (x[2] - x[3])) * 0.5
    return vol
end

function calculate_volume(::Lagrange{3, RefTetrahedron, order}, x::Vector{Vec{3, T}}) where {T, order}
    vol = norm((x[2] - x[1]) ⋅ ((x[3] - x[1]) × (x[4] - x[1]))) / 6.0
    return vol
end

function calculate_volume(::Lagrange{3, RefCube, 1}, x::Vector{Vec{3, T}}) where T
    vol = norm((x[1] - x[5]) ⋅ ((x[2] - x[5]) × (x[4] - x[5]))) / 6.0 +
          norm((x[2] - x[7]) ⋅ ((x[3] - x[7]) × (x[4] - x[7]))) / 6.0 +
          norm((x[2] - x[7]) ⋅ ((x[4] - x[7]) × (x[5] - x[7]))) / 6.0 +
          norm((x[2] - x[7]) ⋅ ((x[5] - x[7]) × (x[6] - x[7]))) / 6.0 +
          norm((x[4] - x[8]) ⋅ ((x[5] - x[8]) × (x[7] - x[8]))) / 6.0
    return vol
end

function calculate_volume(::Serendipity{2, RefCube, 2}, x::Vector{Vec{2, T}}) where T
    vol = norm((x[5] - x[1]) × (x[8] - x[1])) * 0.5 +
          norm((x[6] - x[2]) × (x[5] - x[2])) * 0.5 +
          norm((x[7] - x[3]) × (x[6] - x[3])) * 0.5 +
          norm((x[8] - x[4]) × (x[7] - x[4])) * 0.5 +
          norm((x[6] - x[5]) × (x[8] - x[5])) * 0.5 +
          norm((x[6] - x[7]) × (x[8] - x[7])) * 0.5
    return vol
end

# For faces
function calculate_volume(::Lagrange{0, RefCube, order}, ::Vector{Vec{1, T}}) where {order, T}
    return one(T)
end

coords_on_faces(x, ::Lagrange{1, RefCube, 1}) = ([x[1]], [x[2]])
coords_on_faces(x, ::Lagrange{1, RefCube, 2}) = ([x[1]], [x[2]])
coords_on_faces(x, ::Lagrange{2, RefCube, 1}) =
    ([x[1],x[2]], [x[2],x[3]], [x[3],x[4]], [x[1],x[4]])
coords_on_faces(x, ::Lagrange{2, RefCube, 2}) =
    ([x[1],x[2],x[5]], [x[2],x[3],x[6]], [x[3],x[4],x[7]], [x[1],x[4],x[8]])
coords_on_faces(x, ::Lagrange{2, RefTetrahedron, 1}) =
    ([x[1],x[2]], [x[2],x[3]], [x[1],x[3]])
coords_on_faces(x, ::Lagrange{2, RefTetrahedron, 2}) =
    ([x[1],x[2],x[4]], [x[2],x[3],x[5]], [x[1],x[3],x[6]])
coords_on_faces(x, ::Lagrange{3, RefTetrahedron, 1}) =
    ([x[1],x[2],x[3]], [x[1],x[2],x[4]], [x[2],x[3],x[4]], [x[1],x[3],x[4]])
coords_on_faces(x, ::Lagrange{3, RefTetrahedron, 2}) =
    ([x[1],x[2],x[3],x[5],x[6],x[7]], [x[1],x[2],x[4],x[5],x[8],x[9]], [x[2],x[3],x[4],x[6],x[9],x[10]], [x[1],x[3],x[4],x[7],x[8],x[10]])
coords_on_faces(x, ::Lagrange{3, RefCube, 1}) =
    ([x[1],x[2],x[3],x[4]], [x[1],x[2],x[5],x[6]], [x[2],x[3],x[6],x[7]],[x[3],x[4],x[7],x[8]],[x[1],x[4],x[5],x[8]],[x[5],x[6],x[7],x[8]])
coords_on_faces(x, ::Serendipity{2, RefCube, 2}) =
    ([x[1],x[2],x[5]], [x[2],x[3],x[6]], [x[3],x[4],x[7]], [x[1],x[4],x[8]])
