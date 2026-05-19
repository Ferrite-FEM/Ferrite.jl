"""
    untangle_constraints!(ch::ConstraintHandler)

Untangle the constraints in `ch`. This function will error if the constraints are not tangled.
"""
function untangle_constraints!(ch::ConstraintHandler)
    # Incase user calls this? Maybe not relevant if not exported?
    @assert istangled(ch) "the constraints are not tangled"
    A, affine_equation_ordering, _, dofcoeffs_to_remove = _create_lhs_affine_constraint_matrix(ch)

    # update ch.dofcoefficients so that they can be used to construct `C`
    for (k, v) in dofcoeffs_to_remove
        deleteat!(ch.dofcoefficients[k], v)
    end

    C, g, affine_fdof_ordering = _create_rhs_affine_constraint_matrices(ch, affine_equation_ordering)

    # TODO: maybe add possibility to warn user (if user chooses through kwarg) if `A` is ill conditioned

    luA = try
        LinearAlgebra.lu(A; check = true)
    catch e
        if e isa LinearAlgebra.SingularException
            throw(
                "the affine constraints are tangled and untangling them results in " *
                    "ill defined constraints. A possibility to avoid this is to guarantee that " *
                    "the constraints are not tangled before calling close!"
            )
        else
            rethrow(e)
        end
    end

    C .= LinearAlgebra.ldiv(luA, C)
    _update_dof_coefficients!(ch.dofcoefficients, C, affine_equation_ordering, affine_fdof_ordering)

    # TODO: making affine constraint inhomogeneities time dependent requires (?) saving
    # `A⁻¹` or `luA` as it will be needed in update!
    g .= LinearAlgebra.ldiv(luA, g)

    # we need to update ch.affine_inhomogeneities NOT ch.inhomogeneities
    # as ch.inhomogeneities will be computed in update!
    for (k, v) in affine_equation_ordering
        ch.affine_inhomogeneities[k] = g[v]
    end

    @assert !istangled(ch)
    return ch
end

"""
    _create_lhs_affine_constraint_matrix(ch::ConstraintHandler{DH, T}) where {DH, T}

Create and returns the constraint matrix, `A`, that described the affine
tangled constraints in `ch`. The matrix `A` relates constrained affine dofs, `a_c`, and free, `a_f`, degrees
of freedom via `A * a_c = C * a_f + g`. Three mappings are also returned.

    * `affine_cdof_ordering` which maps `constrained dof => column of A`

    * `affine_equation_ordering` which maps `constraint (eq.) => row of A`

    * `dofcoeffs_to_remove` which maps `constraint (eq.) => position of dof coefficient to remove`

!!! note
    In this case, the system `A * a_c = C * a_f + g` only contains the affine
    constraints that are tangled. Therefore this function is not designed to be used after the
    `ConstraintHandler` has been closed.

"""
function _create_lhs_affine_constraint_matrix(ch::ConstraintHandler{DH, T}) where {DH, T}

    # maps the constrained dofs to a position in `a_c`
    affine_cdof_ordering = Dict{Int, Int}()
    # maps the constraint equation to a row in `A * a_c = C * a_f + g`
    affine_equation_ordering = Dict{Int, Int}()
    # collect the position of the dof coefficients that need to be removed for ch.dofcoefficients
    dofcoeffs_to_remove = Dict{Int, Vector{Int}}()

    I = Int[]; J = Int[]; V = T[]
    dofmapping⁻¹ = Dict{Int, Int}(v => k for (k, v) in ch.dofmapping)

    for (eq, coeffs) in enumerate(ch.dofcoefficients)
        coeffs === nothing && continue # this constraint corresponds to a Dirichlet constraint
        dof_position_counter = 0
        for (d, c) in coeffs
            tangled_eq = get(ch.dofmapping, d, 0)
            dof_position_counter += 1
            tangled_eq == 0 && continue # skip as d is not in the prescribed dofs and therefore not tangled

            tangled_coeffs = ch.dofcoefficients[tangled_eq]
            if !(tangled_coeffs === nothing || isempty(tangled_coeffs)) # nothing means Dirichlet, empty means Dirichlet but through AffineConstraint

                # add the dof to the affine_cdof_ordering
                _assign_new_index!(affine_cdof_ordering, d)
                # add the equation to affine_equation_ordering
                _assign_new_index!(affine_equation_ordering, tangled_eq)

                # add the master dof to affine_cdof_ordering
                _assign_new_index!(affine_cdof_ordering, dofmapping⁻¹[eq])
                # add the equation pertaining to the master dof
                _assign_new_index!(affine_equation_ordering, eq)

                i = affine_equation_ordering[eq]
                j = affine_cdof_ordering[d]
                push!(I, i)
                push!(J, j)
                push!(V, -c)

                # save the position of the dof that needs to be removed
                if !haskey(dofcoeffs_to_remove, eq)
                    dofcoeffs_to_remove[eq] = [dof_position_counter]
                else
                    push!(dofcoeffs_to_remove[eq], dof_position_counter)
                end
            end
        end
    end

    # add the master dof contributions
    for (eq, _) in enumerate(ch.dofcoefficients)
        if haskey(affine_equation_ordering, eq)
            i = affine_equation_ordering[eq]
            j = affine_cdof_ordering[dofmapping⁻¹[eq]]
            push!(I, i)
            push!(J, j)
            push!(V, 1)
        end
    end

    m = length(affine_equation_ordering)
    n = length(affine_cdof_ordering)
    @assert(m == n)

    A = SparseArrays.sparse(I, J, V, m, n)

    return A, affine_equation_ordering, affine_cdof_ordering, dofcoeffs_to_remove
end

"""
    _create_rhs_affine_constraint_matrices(ch::ConstraintHandler{DH, T}, affine_equation_ordering::Dict{Int, Int}) where {DH, T}

Create and return the constraint matrix, `C`, and the inhomogeneities, `g`, from the tangled affine
constraints in `ch`. The constraint matrix relates constrained affine dofs, `a_c`, and free, `a_f`, degrees of freedom via
`A * a_c = C * a_f + g`. A mapping `affine_fdof_ordering` is also returned to index `C`, i.e. it maps "free" dof => column of C.
The rows are indexed using `affine_equation_ordering` from `_create_lhs_affine_constraint_matrix`.

!!! note
    In this case, the system `A * a_c = C * a_f + g` only contains the affine
    constraints that are tangled. Therefore this function is not designed to be used after the
    `ConstraintHandler` has been closed.
"""
function _create_rhs_affine_constraint_matrices(ch::ConstraintHandler{DH, T}, affine_equation_ordering::Dict{Int, Int}) where {DH, T}

    n_tangled_constraints = length(affine_equation_ordering)
    I = Int[]; J = Int[]; V = T[]
    g = Vector{T}(undef, n_tangled_constraints) # inhomogeneities

    # maps the free dofs to a position in `a_f`
    affine_fdof_ordering = Dict{Int, Int}()

    for (eq, coeffs) in enumerate(ch.dofcoefficients)
        (isnothing(coeffs) || !haskey(affine_equation_ordering, eq)) && continue
        i = affine_equation_ordering[eq]
        if isempty(coeffs) && haskey(affine_equation_ordering, eq)
            # the constraint was filled with tangled dofs and now the dof coefficients are empty
            # therefore no contribution in `C` only in `g`
            g[i] = ch.affine_inhomogeneities[eq]
        else
            for (d, v) in coeffs
                _assign_new_index!(affine_fdof_ordering, d)
                j = affine_fdof_ordering[d]
                push!(I, i)
                push!(J, j)
                push!(V, v)
                g[i] = ch.affine_inhomogeneities[eq]
            end
        end
    end

    n = length(affine_fdof_ordering)
    C = SparseArrays.sparse(I, J, V, n_tangled_constraints, n)

    return C, g, affine_fdof_ordering
end

function _update_dof_coefficients!(dc::Vector{Union{Nothing, DofCoefficients{T}}}, C::AbstractMatrix, affine_equation_ordering::Dict{Int, Int}, affine_fdof_ordering::Dict{Int, Int}) where {T}

    affine_fdof_mapping⁻¹ = Dict(v => k for (k, v) in affine_fdof_ordering) # Bijections.jl could avoid this but not really worth it
    affine_equation_ordering⁻¹ = Dict(v => k for (k, v) in affine_equation_ordering)

    for (k, _) in affine_equation_ordering
        dc[k] = DofCoefficients{T}()
    end

    SparseArrays.dropzeros!(C) # make the following loop over C shorter
    for j in axes(C, 2)
        for nz_i in nzrange(C, j)
            i = C.rowval[nz_i]
            dof = affine_fdof_mapping⁻¹[j]
            coeffs = dc[affine_equation_ordering⁻¹[i]]
            push!(coeffs, (dof => C.nzval[nz_i]))
        end
    end
    return dc
end

"""
    istangled(ch::ConstraintHandler)

Check if the constraint handler has any tangled dofs.
"""
function istangled(ch::ConstraintHandler)
    for coeffs in ch.dofcoefficients
        coeffs === nothing && continue
        for (d, _) in coeffs
            i = get(ch.dofmapping, d, 0)
            i == 0 && continue
            icoeffs = ch.dofcoefficients[i]
            if !(icoeffs === nothing || isempty(icoeffs))
                return true
            end
        end
    end
    return false
end

function _assign_new_index!(d::Dict{Int, Int}, k::Int)
    return if !haskey(d, k)
        d[k] = maximum(values(d); init = 0) + 1
    end
end
