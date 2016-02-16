# We specialize (hard code) 2x2 and 3x3 determinants and inverses because
# Julia uses the much slower (but numerically a bit more accurate) result
# that comes from the LU-factorization.
# The tradeoff here is worth it because it is unlikely that we need
# to do computations with almost singular matrices.

# These methods fall back to the normal ones in case the matrix is not 2x2
# or 3x3.
inv_spec(J) = inv_spec!(similar(J), J)

function inv_spec!(Jinv, J)
    dim = chksquare(J)
    @assert size(Jinv) == size(J)
    d = det_spec(J)
    if dim == 1
        @inbounds begin
            Jinv[1,1] = 1/J[1,1]
        end
    if dim == 2
        @inbounds begin
            Jinv[1,1] =  J[2,2] / d
            Jinv[1,2] = -J[1,2] / d
            Jinv[2,1] = -J[2,1] / d
            Jinv[2,2] =  J[1,1] / d
        end
    elseif dim == 3
        @inbounds begin
            Jinv[1,1] =  (J[2,2]*J[3,3] - J[2,3]*J[3,2]) / d
            Jinv[2,1] = -(J[2,1]*J[3,3] - J[2,3]*J[3,1]) / d
            Jinv[3,1] =  (J[2,1]*J[3,2] - J[2,2]*J[3,1]) / d

            Jinv[1,2] = -(J[1,2]*J[3,3] - J[1,3]*J[3,2]) / d
            Jinv[2,2] =  (J[1,1]*J[3,3] - J[1,3]*J[3,1]) / d
            Jinv[3,2] = -(J[1,1]*J[3,2] - J[1,2]*J[3,1]) / d

            Jinv[1,3] =  (J[1,2]*J[2,3] - J[1,3]*J[2,2]) / d
            Jinv[2,3] = -(J[1,1]*J[2,3] - J[1,3]*J[2,1]) / d
            Jinv[3,3] =  (J[1,1]*J[2,2] - J[1,2]*J[2,1]) / d
        end
    else
        Jinv[:,:] = inv(J)
        return Jinv
    end
    return Jinv
end

function det_spec(J)
    dim = chksquare(J)
    d = 0.0
    if dim == 1
        @inbounds d = J[1,1]
    if dim == 2
        @inbounds d = J[1,1]*J[2,2] - J[1,2]*J[2,1]
    elseif dim == 3
        @inbounds begin
            d = J[1,1] * (J[2,2]*J[3,3] - J[2,3]*J[3,2]) -
                J[1,2] * (J[2,1]*J[3,3] - J[2,3]*J[3,1]) +
                J[1,3] * (J[2,1]*J[3,2] - J[2,2]*J[3,1])
        end
    else
        return det(J)
    end
    return d
end
