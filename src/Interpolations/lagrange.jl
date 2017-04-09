############
# Lagrange #
############
immutable Lagrange{dim, shape, order} <: Interpolation{dim, shape, order} end


@pure get_n_vertexdofs{dim, order}(::Lagrange{dim, RefCube, order}) = 1
@pure get_n_edgedofs{dim, order}(::Lagrange{dim, RefCube, order}) = (order - 1)
@pure get_n_surfacedofs{dim, order}(::Lagrange{dim, RefCube, order}) = (order - 1)^(dim - 1)
@pure get_n_celldofs{dim, order}(::Lagrange{dim, RefCube, order}) = (order - 1)^dim
@pure getnbasefunctions{dim, order}(::Lagrange{dim, RefCube, order}) = (order + 1)^dim

@pure get_n_vertexdofs{dim, order}(::Lagrange{dim, RefTetrahedron, order}) = 1
@pure get_n_edgedofs{dim, order}(::Lagrange{dim, RefTetrahedron, order}) = (order - 1)
@pure get_n_surfacedofs{dim, order}(::Lagrange{dim, RefTetrahedron, order}) = error() # TODO
@pure get_n_celldofs{dim, order}(ip::Lagrange{dim, RefTetrahedron, order}) = ((order - 1)^dim - get_n_egedofs(ip)) ÷ 2 # TODO: Check

function get_dof_local_coordinates{order}(ip::Lagrange{1, RefCube, order})
    x = [Vec{1}((x,)) for x in GaussQuadrature.legendre(Float64, order + 1, GaussQuadrature.both)[1]]
    x_reorded = similar(x)
    x_reorded[1] = x[1]
    x_reorded[2] = x[end]
    for i in 2:length(x)-1
      x_reorded[i+1] = x[i]
    end
    return x_reorded
end

function lagrange_polynomial(x::Number, xs::AbstractVector, j::Int)
    @assert j <= length(xs)
    num, den = one(x), one(x)
    @inbounds for i in 1:length(xs)
        i == j && continue
        num *= (x - xs[i])
        den *= (xs[j] - xs[i])
    end
    return num / den
end

function evaluate_Nmatrix{T, order, dim}(ip::Lagrange{dim, RefCube, order}, ξ::AbstractVector{T})
    @assert length(ξ) == dim
    x = GaussQuadrature.legendre(Float64, order + 1, GaussQuadrature.both)[1]
    N =  ones(T, ntuple(i -> 1+order, dim)...)
    for k in 1:dim
        for i in 1:order+1
            # taken from the code in slicedim
            N[( n==k ? i : indices(N, n) for n in 1:ndims(N) )...] .*= lagrange_polynomial(ξ[k], x, i)
        end
    end
    return N
end



function value!{order}(ip::Lagrange{1, RefCube, order}, N::AbstractVector, ξ::AbstractVector)
    checkdim_value(ip, N, ξ)
    Nmat = evaluate_Nmatrix(ip, ξ)
    N[1] = Nmat[1]
    N[2] = Nmat[end]
    for i in 2:length(Nmat)-1
      N[i+1] = Nmat[i]
    end
    return N
end

getnbasefunctions(::Lagrange{2, RefTetrahedron, 1}) = 3

function value!(ip::Lagrange{2, RefTetrahedron, 1}, N::AbstractVector, ξ::AbstractVector)
    checkdim_value(ip, N, ξ)

    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]

        N[1] = ξ_x
        N[2] = ξ_y
        N[3] = 1. - ξ_x - ξ_y
    end

    return N
end

#########################################
# Lagrange dim 2 RefTetrahedron order 2 #
#########################################
getnbasefunctions(::Lagrange{2, RefTetrahedron, 2}) = 6

function value!(ip::Lagrange{2, RefTetrahedron, 2}, N::AbstractVector, ξ::AbstractVector)
    checkdim_value(ip, N, ξ)

    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]

        γ = 1. - ξ_x - ξ_y

        N[1] = ξ_x * (2ξ_x - 1)
        N[2] = ξ_y * (2ξ_y - 1)
        N[3] = γ * (2γ - 1)
        N[4] = 4ξ_x * ξ_y
        N[5] = 4ξ_y * γ
        N[6] = 4ξ_x * γ
    end

    return N
end

#########################################
# Lagrange dim 3 RefTetrahedron order 1 #
#########################################
getnbasefunctions(::Lagrange{3, RefTetrahedron, 1}) = 4

function value!(ip::Lagrange{3, RefTetrahedron, 1}, N::AbstractVector, ξ::AbstractVector)
    checkdim_value(ip, N, ξ)

    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]
        ξ_z = ξ[3]

        N[1] = 1.0 - ξ_x - ξ_y - ξ_z
        N[2] = ξ_x
        N[3] = ξ_y
        N[4] = ξ_z
    end

    return N
end

#########################################
# Lagrange dim 3 RefTetrahedron order 2 #
#########################################
getnbasefunctions(::Lagrange{3, RefTetrahedron, 2}) = 10

# http://www.colorado.edu/engineering/CAS/courses.d/AFEM.d/AFEM.Ch09.d/AFEM.Ch09.pdf
# http://www.colorado.edu/engineering/CAS/courses.d/AFEM.d/AFEM.Ch10.d/AFEM.Ch10.pdf
function value!(ip::Lagrange{3, RefTetrahedron, 2}, N::AbstractVector, ξ::AbstractVector)
    checkdim_value(ip, N, ξ)

    @inbounds begin
        ξ_x = ξ[1]
        ξ_y = ξ[2]
        ξ_z = ξ[3]

        N[1]  = (-2 * ξ_x - 2 * ξ_y - 2 * ξ_z + 1) * (-ξ_x - ξ_y - ξ_z + 1)
        N[2]  = ξ_x * (2 * ξ_x - 1)
        N[3]  = ξ_y * (2 * ξ_y - 1)
        N[4]  = ξ_z * (2 * ξ_z - 1)
        N[5]  = ξ_x * (-4 * ξ_x - 4 * ξ_y - 4 * ξ_z + 4)
        N[6]  = 4 * ξ_x * ξ_y
        N[7]  = 4 * ξ_y * (-ξ_x - ξ_y - ξ_z + 1)
        N[8]  = ξ_z * (-4 * ξ_x - 4 * ξ_y - 4 * ξ_z + 4)
        N[9]  = 4 * ξ_x * ξ_z
        N[10] = 4 * ξ_y * ξ_z
    end

    return N
end
