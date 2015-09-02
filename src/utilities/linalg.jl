function inv2x2(J)
    d = det2x2(J)
    Jinv = zeros(2,2)
    @inbounds begin
        Jinv[1,1] =  J[2,2] / d
        Jinv[1,2] = -J[1,2] / d
        Jinv[2,1] = -J[2,1] / d
        Jinv[2,2] =  J[1,1] / d
    end
    return Jinv
end

function det2x2(A)
    @inbounds d = A[1,1]*A[2,2] - A[1,2]*A[2,1]
    d
end