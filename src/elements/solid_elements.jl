function contmech_lhs_kernel()
    quote
        for i in 1:nnodes
            # Rewrite this with loops instead like for rhs_kernel? /KC
            if ndim == 2
                B[1, 2*i - 1] = dNdx[1, i]
                B[2, 2*i - 0] = dNdx[2, i]
                B[4, 2*i - 0] = dNdx[1, i]
                B[4, 2*i - 1] = dNdx[2, i]
            else
                B[1, i * 3-2] = dNdx[1, i]
                B[2, i * 3-1] = dNdx[2, i]
                B[3, i * 3-0] = dNdx[3, i]
                B[4, 3 * i-1] = dNdx[3, i]
                B[4, 3 * i-0] = dNdx[2, i]
                B[5, 3 * i-2] = dNdx[3, i]
                B[5, 3 * i-0] = dNdx[1, i]
                B[6, 3 * i-2] = dNdx[2, i]
                B[6, 3 * i-1] = dNdx[1, i]
            end
        end
        @into! DB = D * B
        @into! LHS_KERNEL = B' * DB
        if ndim == 2 scale!(LHS_KERNEL, t) end
    end
end

# A RHS kernel should be written such that it sets the variable
# RHS_KERNEL to the left hand
function contmech_rhs_kernel()
    quote
        for i = 1:ndim
            N2[i:ndim:end, i] = N
        end
        @into! RHS_KERNEL = N2 * eq
        if ndim == 2 scale!(RHS_KERNEL, t) end
    end
end

# Returns the commonly needed variables for a
# continuum mechanics problem.
function get_default_contmech_vars(nnodes, ndim)
    ndofs = nnodes * ndim
    if ndim == 2
        nvars = 4 # plain strain/stress
    elseif ndim == 3
        nvars = 6
    else
        throw(ArgumentError("Invalid dimension, must be 2, 3"))
    end
    Dict(:DB => (nvars,ndofs),
         :N2 => (ndofs,ndim),
         :B => (nvars,ndofs))
end


S_S_1 = FElement(
    :solid_square_1,
    Square(),
    Lagrange_S_1(),
    get_default_contmech_vars(4, 2),
    4,
    2,
    contmech_lhs_kernel,
    contmech_rhs_kernel,
    2
)


S_S_2 = FElement(
    :solid_square_2,
    Square(),
    Lagrange_S_2(),
    get_default_contmech_vars(8, 2),
    8,
    2,
    contmech_lhs_kernel,
    contmech_rhs_kernel,
    3
)


S_T_1 = FElement(
    :solid_tri_1,
    Triangle(),
    Lagrange_T_1(),
    get_default_contmech_vars(3, 2),
    3,
    2,
    contmech_lhs_kernel,
    contmech_rhs_kernel,
    1
)


S_C_1 = FElement(
    :solid_cube_1,
    Cube(),
    Lagrange_C_1(),
    get_default_contmech_vars(8, 3),
    8,
    3,
    contmech_lhs_kernel,
    contmech_rhs_kernel,
    2
)
