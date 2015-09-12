# We specialize (hard code) 2x2 and 3x3 determinants and inverses because
# Julia uses the much slower (but numerically a bit more accurate) result
# that comes from the LU-factorization.
# The tradeoff here is worth it because it is unlikely that we need
# to do computations with almost singular matrices.

# These methods fall back to the normal ones in case the matrix is not 2x2
# or 3x3.

function inv_spec(J)
    dim = chksquare(J)
    d = det_spec(J)
    Jinv = zeros(dim, dim)
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
        return inv(J)
    end
    return Jinv
end

function det_spec(J)
    dim = chksquare(J)
    d = 0.0
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
