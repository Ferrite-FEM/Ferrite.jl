# Some utility functions for testing Ferrite.jl

#####################################
# Volume for the reference elements #
#####################################
reference_volume(::Interpolation{Ferrite.RefHypercube{dim}}) where {dim} = 2^dim
reference_volume(::Interpolation{Ferrite.RefSimplex{dim}}) where {dim} = 1 / factorial(dim)
reference_volume(::Interpolation{RefPrism}) = 1/2
# For faces
reference_face_area(fs::VectorizedInterpolation, f::Int) = reference_face_area(fs.ip, f)
reference_face_area(fs::Interpolation{Ferrite.RefHypercube{dim}}, face::Int) where {dim} = 2^(dim-1)
reference_face_area(fs::Interpolation{RefTriangle}, face::Int) = face == 1 ? sqrt(2) : 1.0
reference_face_area(fs::Interpolation{RefTetrahedron}, face::Int) = face == 3 ? sqrt(2 * 1.5) / 2.0 : 0.5

######################################################
# Coordinates and normals for the reference elements #
######################################################

reference_normals(ip::VectorizedInterpolation) = reference_normals(ip.ip)

# Lagrange{1, RefLine}
function reference_normals(::Lagrange{RefLine})
    return [Vec{1, Float64}((-1.0,)),
            Vec{1, Float64}(( 1.0,))]
end

# Lagrange{2, RefQuadrilateral}
function reference_normals(::Lagrange{RefQuadrilateral})
    return [Vec{2, Float64}(( 0.0, -1.0)),
            Vec{2, Float64}(( 1.0,  0.0)),
            Vec{2, Float64}(( 0.0,  1.0,)),
            Vec{2, Float64}((-1.0,  0.0,))]
end

# Lagrange{2, RefTriangle}
function reference_normals(::Lagrange{RefTriangle})
    return [Vec{2, Float64}((1/√2, 1/√2)),
            Vec{2, Float64}((-1.0, 0.0)),
            Vec{2, Float64}((0.0, -1.0))]
end

# Lagrange{3, RefTetrahedron}
function reference_normals(::Lagrange{RefTetrahedron})
    return [Vec{3, Float64}((0.0, 0.0, -1.0)),
            Vec{3, Float64}((0.0 ,-1.0, 0.0)),
            Vec{3, Float64}((1/√3, 1/√3, 1/√3)),
            Vec{3, Float64}((-1.0, 0.0, 0.0))]
end

# Lagrange{3, Cube}
function reference_normals(::Lagrange{RefHexahedron})
    return [Vec{3, Float64}(( 0.0,  0.0, -1.0)),
            Vec{3, Float64}(( 0.0, -1.0,  0.0)),
            Vec{3, Float64}(( 1.0,  0.0,  0.0)),
            Vec{3, Float64}(( 0.0,  1.0,  0.0)),
            Vec{3, Float64}((-1.0,  0.0,  0.0)),
            Vec{3, Float64}(( 0.0,  0.0,  1.0))]
end

# Lagrange{3, Wedge}
function reference_normals(::Lagrange{RefPrism})
    return [Vec{3, Float64}(( 0.0,  0.0, -1.0)),
            Vec{3, Float64}(( 0.0, -1.0,  0.0)),
            Vec{3, Float64}((-1.0,  0.0,  0.0)),
            Vec{3, Float64}((1/√2, 1/√2,  0.0)),
            Vec{3, Float64}(( 0.0,  0.0,  1.0))]
end

# Lagrange{3, Wedge}
function reference_normals(::Lagrange{RefPyramid})
    return [Vec{3, Float64}(( 0.0,  0.0, -1.0)),
            Vec{3, Float64}(( 0.0, -1.0,  0.0)),
            Vec{3, Float64}((1/√2, 0.0,  1/√2)),
            Vec{3, Float64}((0.0, 1/√2,  1/√2)),
            Vec{3, Float64}((-1.0,  0.0,  0.0)),]
end

# Serendipity{2, RefQuadrilateral}
reference_normals(::Serendipity{RefQuadrilateral, 2}) = reference_normals(Lagrange{RefQuadrilateral, 1}())

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

function valid_coordinates_and_normals(fs::Interpolation{shape, order}) where {dim, shape <: Ferrite.AbstractRefShape{dim}, order}
    x = Ferrite.reference_coordinates(fs)
    n = reference_normals(fs)
    R = rotmat(dim)
    return [2.0 * (R ⋅ x[i]) for i in 1:length(x)] , [(R ⋅ n[i]) / norm((R ⋅ n[i])) for i in 1:length(n)]
end

#######################################
# Volume of cells (with planar edges) #
#######################################

calculate_volume(ip::VectorizedInterpolation, x) = calculate_volume(ip.ip, x)

function calculate_volume(::Lagrange{RefLine, 1}, x::Vector{Vec{dim, T}}) where {T, dim}
    vol = norm(x[2] - x[1])
    return vol
end

function calculate_volume(::Lagrange{RefLine, 2}, x::Vector{Vec{dim, T}}) where {T, dim}
    vol = norm(x[3] - x[1]) + norm(x[2]-x[3])
    return vol
end

function calculate_volume(::Lagrange{RefQuadrilateral, 1}, x::Vector{Vec{dim, T}}) where {T, dim}
    vol = norm((x[4] - x[1]) × (x[2] - x[1])) * 0.5 +
          norm((x[4] - x[3]) × (x[2] - x[3])) * 0.5
    return vol
end

function calculate_volume(::Lagrange{RefQuadrilateral, 2}, x::Vector{Vec{dim, T}}) where {T, dim}
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

function calculate_volume(::Lagrange{RefTriangle, 1}, x::Vector{Vec{dim, T}}) where {T, dim}
    vol = norm((x[1] - x[3]) × (x[2] - x[3])) * 0.5
    return vol
end

function calculate_volume(::Lagrange{RefTriangle, 2}, x::Vector{Vec{dim, T}}) where {T, dim}
    vol = norm((x[6] - x[3]) × (x[5] - x[3])) * 0.5 +
          norm((x[6] - x[4]) × (x[5] - x[4])) * 0.5 +
          norm((x[1] - x[6]) × (x[4] - x[6])) * 0.5 +
          norm((x[4] - x[5]) × (x[2] - x[5])) * 0.5
    return vol
end

# TODO: Only correct for linear sides
function calculate_volume(::Lagrange{RefTriangle, O}, x::Vector{Vec{dim, T}}) where {T, dim, O}
    vol = norm((x[1] - x[3]) × (x[2] - x[3])) * 0.5
    return vol
end

function calculate_volume(::Lagrange{RefTetrahedron, order}, x::Vector{Vec{3, T}}) where {T, order}
    vol = norm((x[2] - x[1]) ⋅ ((x[3] - x[1]) × (x[4] - x[1]))) / 6.0
    return vol
end

function calculate_volume(::Lagrange{RefHexahedron, 1}, x::Vector{Vec{3, T}}) where T
    vol = norm((x[1] - x[5]) ⋅ ((x[2] - x[5]) × (x[4] - x[5]))) / 6.0 +
          norm((x[2] - x[7]) ⋅ ((x[3] - x[7]) × (x[4] - x[7]))) / 6.0 +
          norm((x[2] - x[7]) ⋅ ((x[4] - x[7]) × (x[5] - x[7]))) / 6.0 +
          norm((x[2] - x[7]) ⋅ ((x[5] - x[7]) × (x[6] - x[7]))) / 6.0 +
          norm((x[4] - x[8]) ⋅ ((x[5] - x[8]) × (x[7] - x[8]))) / 6.0
    return vol
end

function calculate_volume(::Serendipity{RefQuadrilateral, 2}, x::Vector{Vec{2, T}}) where T
    vol = norm((x[5] - x[1]) × (x[8] - x[1])) * 0.5 +
          norm((x[6] - x[2]) × (x[5] - x[2])) * 0.5 +
          norm((x[7] - x[3]) × (x[6] - x[3])) * 0.5 +
          norm((x[8] - x[4]) × (x[7] - x[4])) * 0.5 +
          norm((x[6] - x[5]) × (x[8] - x[5])) * 0.5 +
          norm((x[6] - x[7]) × (x[8] - x[7])) * 0.5
    return vol
end

function calculate_face_area(ip::Lagrange{RefLine}, x::Vector{<:Vec}, faceindex::Int)
    return one(eltype(eltype(x)))
end
function calculate_face_area(ip::Lagrange{RefQuadrilateral, order}, x::Vector{<:Vec}, faceindex::Int) where order
    return calculate_volume(Lagrange{RefLine, order}(), x)
end
function calculate_face_area(ip::Lagrange{RefTriangle, order}, x::Vector{<:Vec}, faceindex::Int) where order
    return calculate_volume(Lagrange{RefLine, order}(), x)
end
function calculate_face_area(ip::Lagrange{RefHexahedron, order}, x::Vector{<:Vec}, faceindex::Int) where order
    return calculate_volume(Lagrange{RefQuadrilateral, order}(), x)
end
function calculate_face_area(ip::Serendipity{RefQuadrilateral, order}, x::Vector{<:Vec}, faceindex::Int) where order
    return calculate_volume(Lagrange{RefLine, order}(), x)
end
function calculate_face_area(ip::Lagrange{RefTetrahedron, order}, x::Vector{<:Vec}, faceindex::Int) where order
    return calculate_volume(Lagrange{RefTriangle, order}(), x)
end

coords_on_faces(x, ::Lagrange{RefLine, 1}) = ([x[1]], [x[2]])
coords_on_faces(x, ::Lagrange{RefLine, 2}) = ([x[1]], [x[2]])
coords_on_faces(x, ::Lagrange{RefQuadrilateral, 1}) =
    ([x[1],x[2]], [x[2],x[3]], [x[3],x[4]], [x[1],x[4]])
coords_on_faces(x, ::Lagrange{RefQuadrilateral, 2}) =
    ([x[1],x[2],x[5]], [x[2],x[3],x[6]], [x[3],x[4],x[7]], [x[1],x[4],x[8]])
coords_on_faces(x, ::Lagrange{RefTriangle, 1}) =
    ([x[1],x[2]], [x[2],x[3]], [x[1],x[3]])
coords_on_faces(x, ::Lagrange{RefTriangle, 2}) =
    ([x[1],x[2],x[4]], [x[2],x[3],x[5]], [x[1],x[3],x[6]])
coords_on_faces(x, ::Lagrange{RefTetrahedron, 1}) =
    ([x[1],x[2],x[3]], [x[1],x[2],x[4]], [x[2],x[3],x[4]], [x[1],x[3],x[4]])
coords_on_faces(x, ::Lagrange{RefTetrahedron, 2}) =
    ([x[1],x[2],x[3],x[5],x[6],x[7]], [x[1],x[2],x[4],x[5],x[8],x[9]], [x[2],x[3],x[4],x[6],x[9],x[10]], [x[1],x[3],x[4],x[7],x[8],x[10]])
coords_on_faces(x, ::Lagrange{RefHexahedron, 1}) =
    ([x[1],x[2],x[3],x[4]], [x[1],x[2],x[5],x[6]], [x[2],x[3],x[6],x[7]],[x[3],x[4],x[7],x[8]],[x[1],x[4],x[5],x[8]],[x[5],x[6],x[7],x[8]])
coords_on_faces(x, ::Serendipity{RefHexahedron, 2}) =
    ([x[1],x[2],x[5]], [x[2],x[3],x[6]], [x[3],x[4],x[7]], [x[1],x[4],x[8]])
