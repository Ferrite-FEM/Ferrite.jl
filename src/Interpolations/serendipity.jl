###############
# Serendipity #
###############
immutable Serendipity{dim, shape, order} <: Interpolation{dim, shape, order} end

#####################################
# Serendipity dim 2 RefCube order 2 #
#####################################
getnbasefunctions(::Serendipity{2, RefCube, 2}) = 8


function value!(ip::Serendipity{2, RefCube, 2}, N::AbstractVector, ξ::AbstractVector)
    checkdim_value(ip, N, ξ)

    ξ_x = ξ[1]
    ξ_y = ξ[2]

    @inbounds begin
        N[1] = (1 - ξ_x) * (1 - ξ_y) * 0.25(-ξ_x - ξ_y - 1)
        N[2] = (1 + ξ_x) * (1 - ξ_y) * 0.25( ξ_x - ξ_y - 1)
        N[3] = (1 + ξ_x) * (1 + ξ_y) * 0.25( ξ_x + ξ_y - 1)
        N[4] = (1 - ξ_x) * (1 + ξ_y) * 0.25(-ξ_x + ξ_y - 1)
        N[6] = 0.5(1 + ξ_x) * (1 - ξ_y * ξ_y)
        N[5] = 0.5(1 - ξ_x * ξ_x) * (1 - ξ_y)
        N[7] = 0.5(1 - ξ_x * ξ_x) * (1 + ξ_y)
        N[8] = 0.5(1 - ξ_x) * (1 - ξ_y * ξ_y)
    end
    return N
end
