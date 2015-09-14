function heat_grad_kernel()
    quote
        @into! DB = D * dNdx
        @into! GRAD_KERNEL = dNdx' * DB
        if ndim == 2 scale!(GRAD_KERNEL, t) end
    end
end

# A RHS kernel should be written such that it sets the variable
# SOURCE_KERNEL to the left hand
function heat_source_kernel()
    quote
        @into! SOURCE_KERNEL = N * eq
        if ndim == 2 scale!(SOURCE_KERNEL, t) end
    end
end

function get_default_heat_vars(nnodes, ndim)
    Dict(:DB => (ndim ,nnodes))
end

H_S_1 = FElement(
    :heat_square_1,
    Square(),
    Lagrange_Square_1(),
    get_default_heat_vars(4, 2),
    4,
    1,
    heat_grad_kernel,
    heat_source_kernel,
    2)

H_S_2 = FElement(
    :heat_square_2,
    Square(),
    Serend_Square_2(),
    get_default_heat_vars(8, 2),
    8,
    1,
    heat_grad_kernel,
    heat_source_kernel,
    3)

H_T_1 = FElement(
    :heat_tri_1,
    Triangle(),
    Lagrange_Tri_1(),
    get_default_heat_vars(3, 2),
    3,
    1,
    heat_grad_kernel,
    heat_source_kernel,
    1)

H_C_1 = FElement(
    :heat_cube_1,
    Cube(),
    Lagrange_Cube_1(),
    get_default_heat_vars(8, 3),
    8,
    1,
    heat_grad_kernel,
    heat_source_kernel,
    2)
