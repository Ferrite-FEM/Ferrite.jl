function Bmatrix!(B, nnodes, ndim, dNdx)
     for i in 1:nnodes
        dNdXi = dNdx[i]
        # Rewrite this with loops instead like for source_kernel? /KC
        if ndim == 2
            @inbounds begin
                B[1, 2*i - 1] = dNdXi[1]
                B[2, 2*i - 0] = dNdXi[2]
                B[4, 2*i - 0] = dNdXi[1]
                B[4, 2*i - 1] = dNdXi[2]
            end
        else
            @inbounds begin
                B[1, i * 3-2] = dNdXi[1]
                B[2, i * 3-1] = dNdXi[2]
                B[3, i * 3-0] = dNdXi[3]
                B[4, 3 * i-1] = dNdXi[3]
                B[4, 3 * i-0] = dNdXi[2]
                B[5, 3 * i-2] = dNdXi[3]
                B[5, 3 * i-0] = dNdXi[1]
                B[6, 3 * i-2] = dNdXi[2]
                B[6, 3 * i-1] = dNdXi[1]
            end
        end
    end
end

function contmech_grad_kernel()
    quote
        Bmatrix!(B, nnodes, ndim, dNdx)
        @into! DB = D * B
        @into! GRAD_KERNEL = B' * DB
        if ndim == 2 scale!(GRAD_KERNEL, t) end
    end
end

# A RHS kernel should be written such that it sets the variable
# SOURCE_KERNEL to the left hand
function contmech_source_kernel()
    quote
        for i = 1:ndim
            N2[i:ndim:end, i] = N
        end
        @into! SOURCE_KERNEL = N2 * eq
        if ndim == 2 scale!(SOURCE_KERNEL, t) end
    end
end

# A flux kernel should be written such that it sets the variable
# FLUX_KERNEL and CONJ_KERNEL
function contmech_flux_kernel()
    quote
        Bmatrix!(B, nnodes, ndim, dNdx)
        @into! CONJ_KERNEL = B * ed
        @into! FLUX_KERNEL = D * CONJ_KERNEL
    end
end

# An intf kernel should be written such that it sets the variable
# INTF
function contmech_intf_kernel()
    quote
        Bmatrix!(B, nnodes, ndim, dNdx)
        @into! INTF_KERNEL = B' * σ
        if ndim == 2 scale!(INTF_KERNEL, t) end
    end
end


function get_contmech_flux_size(ndim)
    if ndim == 2
        return 4
    else
        return 6
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
    Lagrange{2, Square, 1}(),
    get_default_contmech_vars(4, 2),
    4,
    2,
    get_contmech_flux_size(2),
    contmech_grad_kernel,
    contmech_source_kernel,
    contmech_flux_kernel,
    contmech_intf_kernel,
    2
)


S_S_2 = FElement(
    :solid_square_2,
    Serendipity{2, Square, 2}(),
    get_default_contmech_vars(8, 2),
    8,
    2,
    get_contmech_flux_size(2),
    contmech_grad_kernel,
    contmech_source_kernel,
    contmech_flux_kernel,
    contmech_intf_kernel,
    3
)


S_T_1 = FElement(
    :solid_tri_1,
    Lagrange{2, Triangle, 1}(),
    get_default_contmech_vars(3, 2),
    3,
    2,
    get_contmech_flux_size(2),
    contmech_grad_kernel,
    contmech_source_kernel,
    contmech_flux_kernel,
    contmech_intf_kernel,
    1
)


S_C_1 = FElement(
    :solid_cube_1,
    Lagrange{3, Square, 1}(),
    get_default_contmech_vars(8, 3),
    8,
    3,
    get_contmech_flux_size(3),
    contmech_grad_kernel,
    contmech_source_kernel,
    contmech_flux_kernel,
    contmech_intf_kernel,
    2
)
