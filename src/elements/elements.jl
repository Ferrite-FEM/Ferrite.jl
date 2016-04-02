# Classic structural elements
include("CALFEM_API.jl")
include("CALFEM_docs.jl")
include("spring.jl")
include("bar.jl")

include("solid_elements.jl")
include("heat_elements.jl")

create_initial_vars(inits) = Expr(:block, [:($sym = zeros($size)) for (sym, size) in inits]...)

"""
This function generates the body expression for a function that
computes the so called stiffness matrix and force vector
for a finite element.

The finite element should be an instance of `FElement`.
"""
function gen_ke_fe_body(ele)
    quote
        ndofs = $(ele.nnodes) * $(ele.dofs_per_node)
        nnodes = $(ele.nnodes)
        ndim = $(n_dim(ele))
        n_basefuncs = n_basefunctions($(ele.function_space))
        @assert length(x) == n_basefuncs * ndim
        xvec = reinterpret(Vec{$(n_dim(ele)), Float64}, x, (n_basefuncs,))
        # TODO; need to fix the eq size for different problems
        int_order > 0 || throw(ArgumentError("integration order must be > 0"))

        # Default buffers
        Ke = zeros(ndofs, ndofs)
        fe = zeros(ndofs)
        GRAD_KERNEL = zeros(ndofs, ndofs)
        SOURCE_KERNEL = zeros(ndofs)
        dNdξ = [zero(Vec{$(n_dim(ele)), Float64}) for i in 1:n_basefuncs]
        dNdx = [zero(Vec{$(n_dim(ele)), Float64}) for i in 1:n_basefuncs]
        N = zeros(nnodes)

        # Create the element requested buffers
        $(create_initial_vars(ele.inits))

        # Check if we need to compute the RHS
        compute_RHS = (eq != zeros(ndim))

        # Create the gauss rule of the requested order on the
        # elements reference shape
        qr = get_gaussrule(Dim{$(n_dim(ele))}, $(ref_shape(ele.function_space)), int_order)

        for (i, (ξ, w)) in enumerate(zip(qr.points, qr.weights))
            value!($(ele.function_space), N, ξ)
            derivative!($(ele.function_space), dNdξ, ξ)
            J = zero(Tensor{2, $(n_dim(ele))})
            for j in 1:n_basefuncs
                J += dNdξ[j] ⊗ xvec[j]
            end
            Jinv = inv(J)
            for j in 1:n_basefuncs
                dNdx[j] = Jinv ⋅ dNdξ[j]
            end
            dV = det(J) * w

            ##############################
            # Call the elements LHS kernel
            $(ele.grad_kernel())
            ##############################

            @devec Ke[:, :] += GRAD_KERNEL .* dV

            if compute_RHS
                ##############################
                # Call the elements RHS kernel
                $(ele.source_kernel())
                ##############################

                @devec fe += SOURCE_KERNEL .* dV
            end
        end
        return Ke, fe
    end
end

for fem in [S_S_1, S_S_2, S_T_1, S_C_1, # Solid elements
            H_S_1, H_S_2, H_T_1, H_C_1]
    f_name = Symbol(string(fem.name) * "e")
    if n_dim(fem) == 2
        @eval function $(f_name)(x::Matrix, D::Matrix, t::Number,
                                  eq::VecOrMat=zeros($(n_dim(fem))),
                                  int_order::Int=$(fem.default_intorder))
            $(gen_ke_fe_body(fem))
        end
    elseif n_dim(fem) == 3
        @eval function $(f_name)(x::Matrix, D::Matrix,
                                  eq::VecOrMat=zeros($(n_dim(fem))),
                                  int_order::Int=$(fem.default_intorder))
            $(gen_ke_fe_body(fem))
        end
    end
end


"""
This function generates the body expression for a function that
computes the flux (stress) and its conjugate(strain)

The finite element should be an instance of `FElement`.
"""
function gen_s_body(ele)
    quote
        ndofs = $(ele.nnodes) * $(ele.dofs_per_node)
        nnodes = $(ele.nnodes)
        ndim = $(n_dim(ele))
        n_basefuncs = n_basefunctions($(ele.function_space))
        @assert length(x) == n_basefuncs * ndim
        xvec = reinterpret(Vec{$(n_dim(ele)), Float64}, x, (n_basefuncs,))
        fluxlen = $(n_flux(ele))

        #length(eq) == ndim || throw(ArgumentError("length of eq must be $ndim"))
        int_order > 0 || throw(ArgumentError("integration order must be > 0"))

        # Default buffers
        FLUX_KERNEL = zeros(fluxlen)
        CONJ_KERNEL = zeros(fluxlen)
        dNdξ = [zero(Vec{$(n_dim(ele)), Float64}) for i in 1:n_basefuncs]
        dNdx = [zero(Vec{$(n_dim(ele)), Float64}) for i in 1:n_basefuncs]
        N = zeros(nnodes)

        # Create the element requested buffers
        $(create_initial_vars(ele.inits))

        # Create the gauss rule of the requested order on the
        # elements reference shape
        qr = get_gaussrule(Dim{$(n_dim(ele))}, $(ref_shape(ele.function_space)), int_order)

        n_gps = length(qr.points)
        FLUXES = zeros(fluxlen, n_gps)
        CONJS = zeros(fluxlen, n_gps)
        points = [zero(Vec{$(n_dim(ele)), Float64}) for j in 1:length(qr.points)]

        for (i, (ξ, w)) in enumerate(zip(qr.points, qr.weights))
            value!($(ele.function_space), N, ξ)
            derivative!($(ele.function_space), dNdξ, ξ)
            J = zero(Tensor{2, $(n_dim(ele))})
            for j in 1:n_basefuncs
                J += dNdξ[j] ⊗ xvec[j]
            end
             Jinv = inv(J)
            for j in 1:n_basefuncs
                dNdx[j] = Jinv ⋅ dNdξ[j]
            end
            dV = det(J) * w
            ##############################
            # Call the elements flux kernel
            $(ele.flux_kernel())
            ##############################

            FLUXES[:, i] = FLUX_KERNEL
            CONJS[:, i] = CONJ_KERNEL
            for j in 1:n_basefuncs
                points[i] += xvec[j] * N[j]
            end
        end

        return FLUXES, CONJS, reinterpret(Float64, points, ($(n_dim(ele)) *length(qr.points),))
    end
end

for fem in [S_S_1, S_S_2, S_T_1, S_C_1, # Solid elements
            H_S_1, H_S_2, H_T_1, H_C_1]
    f_name = Symbol(string(fem.name) * "s")
    if n_dim(fem) == 2
        @eval function $(f_name)(x::Matrix, D::Matrix, t::Number,
                                 ed::VecOrMat,
                                 int_order::Int=$(fem.default_intorder))
            $(gen_s_body(fem))
        end
    elseif n_dim(fem) == 3
        @eval function $(f_name)(x::Matrix, D::Matrix,
                                 ed::VecOrMat,
                                 int_order::Int=$(fem.default_intorder))
            $(gen_s_body(fem))
        end
    end
end


"""
This function generates the body expression for a function that
computes the internal forces.

The finite element should be an instance of `FElement`.
"""
function gen_f_body(ele)
    quote
        ndofs = $(ele.nnodes) * $(ele.dofs_per_node)
        nnodes = $(ele.nnodes)
        ndim = $(n_dim(ele))
        n_basefuncs = n_basefunctions($(ele.function_space))
        @assert length(x) == n_basefuncs * ndim
        xvec = reinterpret(Vec{$(n_dim(ele)), Float64}, x, (n_basefuncs,))
        fluxlen = $(n_flux(ele))

        #length(eq) == ndim || throw(ArgumentError("length of eq must be $ndim"))
        int_order > 0 || throw(ArgumentError("integration order must be > 0"))

        # Default buffers
        intf = zeros(ndofs)
        INTF_KERNEL = zeros(ndofs)
        dNdξ = [zero(Vec{$(n_dim(ele)), Float64}) for i in 1:n_basefuncs]
        dNdx = [zero(Vec{$(n_dim(ele)), Float64}) for i in 1:n_basefuncs]
        N = zeros(nnodes)

        # Create the element requested buffers
        $(create_initial_vars(ele.inits))

        # Create the gauss rule of the requested order on the
        # elements reference shape
        qr = get_gaussrule(Dim{$(n_dim(ele))}, $(ref_shape(ele.function_space)), int_order)

        if size(σs, 2) != length(qr.points)
            throw(ArgumentError("must use same integration rule to compute stresses and internal forces"))
        end

        for (i, (ξ, w)) in enumerate(zip(qr.points, qr.weights))
            value!($(ele.function_space), N, ξ)
            derivative!($(ele.function_space), dNdξ, ξ)
            J = zero(Tensor{2, $(n_dim(ele))})
            for j in 1:n_basefuncs
                J += dNdξ[j] ⊗ xvec[j]
            end
            Jinv = inv(J)
            for j in 1:n_basefuncs
                dNdx[j] = Jinv ⋅ dNdξ[j]
            end
            dV = det(J) * w
            σ = vec(σs[:, i])

            ##############################
            # Call the elements flux kernel
            $(ele.intf_kernel())
            ##############################

            @devec intf[:] += INTF_KERNEL .* dV
        end
        return intf
    end
end


for fem in [S_S_1, S_S_2, S_T_1, S_C_1] # Only for solid elements
    f_name = Symbol(string(fem.name) * "f")
    if n_dim(fem) == 2
        @eval function $(f_name)(x::Matrix, t::Number, σs::VecOrMat,
                                 int_order::Int=$(fem.default_intorder))
            $(gen_f_body(fem))
        end
    elseif n_dim(fem) == 3
        @eval function $(f_name)(x::Matrix, σs::VecOrMat,
                                 int_order::Int=$(fem.default_intorder))
            $(gen_f_body(fem))
        end
    end
end
