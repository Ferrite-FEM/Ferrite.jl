# Some utility functions for testing JuAFEM.jl

#####################################
# Volume for the reference elements #
#####################################
reference_volume{dim}(::FunctionSpace{dim, RefCube}) = 2^dim
reference_volume{dim}(::FunctionSpace{dim, RefTetrahedron}) = 1 / factorial(dim)
# For boundaries
reference_volume(fs::FunctionSpace, ::Int) = reference_volume(JuAFEM.functionspace_lower_dim(fs))
reference_volume(fs::FunctionSpace{2, RefTetrahedron}, boundary::Int) = boundary == 1 ? sqrt(2) : 1.0
reference_volume(fs::FunctionSpace{3, RefTetrahedron}, b::Int) = b == 3 ? sqrt(2 * 1.5) / 2.0 : 0.5

##########################################
# Coordinates for the reference elements #
##########################################
function reference_coordinates(fs::Lagrange{1, RefCube, 1})
    return [Vec{1, Float64}((-1.0,)),
            Vec{1, Float64}(( 1.0,))]
end

function reference_coordinates(fs::Lagrange{1, RefCube, 2})
    return [Vec{1, Float64}((-1.0,)),
            Vec{1, Float64}(( 0.0,)),
            Vec{1, Float64}(( 1.0,))]
end

function reference_coordinates(fs::Lagrange{2, RefCube, 1})
    return [Vec{2, Float64}((-1.0, -1.0)),
            Vec{2, Float64}(( 1.0, -1.0)),
            Vec{2, Float64}(( 1.0,  1.0,)),
            Vec{2, Float64}((-1.0,  1.0,))]
end

function reference_coordinates(fs::Lagrange{2, RefCube, 2})
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

function reference_coordinates(fs::Lagrange{2, RefTetrahedron, 1})
    return [Vec{2, Float64}((1.0, 0.0)),
            Vec{2, Float64}((0.0, 1.0)),
            Vec{2, Float64}((0.0, 0.0))]
end

function reference_coordinates(fs::Lagrange{2, RefTetrahedron, 2})
    return [Vec{2, Float64}((1.0, 0.0)),
            Vec{2, Float64}((0.0, 1.0)),
            Vec{2, Float64}((0.0, 0.0)),
            Vec{2, Float64}((0.5, 0.5)),
            Vec{2, Float64}((0.0, 0.5)),
            Vec{2, Float64}((0.5, 0.0))]
end

function reference_coordinates(fs::Lagrange{3, RefTetrahedron, 1})
    return [Vec{3, Float64}((1.0, 0.0, 0.0)),
            Vec{3, Float64}((0.0, 1.0, 0.0)),
            Vec{3, Float64}((0.0, 0.0, 1.0)),
            Vec{3, Float64}((0.0, 0.0, 0.0))]
end

function reference_coordinates(fs::Lagrange{3, RefCube, 1})
    return [Vec{3, Float64}((-1.0, -1.0, -1.0)),
            Vec{3, Float64}(( 1.0, -1.0, -1.0)),
            Vec{3, Float64}(( 1.0,  1.0, -1.0)),
            Vec{3, Float64}((-1.0,  1.0, -1.0)),
            Vec{3, Float64}((-1.0, -1.0,  1.0)),
            Vec{3, Float64}(( 1.0, -1.0,  1.0)),
            Vec{3, Float64}(( 1.0,  1.0,  1.0)),
            Vec{3, Float64}((-1.0,  1.0,  1.0))]
end

function reference_coordinates(fs::Serendipity{2, RefCube, 2})
    return [Vec{2, Float64}((-1.0, -1.0)),
            Vec{2, Float64}(( 1.0, -1.0)),
            Vec{2, Float64}(( 1.0,  1.0)),
            Vec{2, Float64}((-1.0,  1.0)),
            Vec{2, Float64}(( 0.0, -1.0)),
            Vec{2, Float64}(( 1.0,  0.0)),
            Vec{2, Float64}(( 0.0,  1.0)),
            Vec{2, Float64}((-1.0,  0.0))]
end

function valid_nodes(fs::FunctionSpace)
    x = reference_coordinates(fs)
    return [x[i] + 0.1 * rand(typeof(x[i])) for i in 1:length(x)]
end
