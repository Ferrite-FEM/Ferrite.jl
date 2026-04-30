function _assign_new_index!(d::Dict{Int,Int}, k::Int)
    if !haskey(d, k)
        d[k] = maximum(values(d); init=0) + 1
    end
end

"""
    _create_lhs_affine_constraint_matrix(ch::ConstraintHandler{DH,T}) where {DH,T}

Create and returns the constraint matrix, `A`, that described the affine 
nested constraints in `ch`. The matrix `A` relates constrained affine dofs, `a_c`, and free, `a_f`, degrees 
of freedom via `A * a_c = C * a_f + g`. Three mappings are also returned.

    * `affine_cdof_ordering` which maps `constrained dof => column of A`

    * `affine_equation_ordering` which maps `constraint (eq.) => row of A`

    * `dofcoeffs_to_remove` which maps `constraint (eq.) => postion of dof coefficient to remove`

!!! note
    In this case, the system `A * a_c = C * a_f + g` only contains the affine 
    constraints that are nested. Therefore this function is not designed to be used after the
    `ConstraintHandler` has been closed.

"""
function _create_lhs_affine_constraint_matrix(ch::ConstraintHandler{DH,T}) where {DH,T}
    
    # maps the constrained dofs to a position in `a_c`
    affine_cdof_ordering = Dict{Int,Int}()
    # maps the constraint equation to a row in `A * a_c = C * a_f + g`
    affine_equation_ordering = Dict{Int,Int}()
    # collect the position of the dof coefficients that need to be removed for ch.dofcoefficients
    dofcoeffs_to_remove = Dict{Int,Vector{Int}}()

    I = Int[]; J = Int[]; V = T[]
    dofmapping⁻¹ = Dict{Int,Int}(v => k for (k, v) in ch.dofmapping)

    for (eq, coeffs) in enumerate(ch.dofcoefficients)
        coeffs === nothing && continue # this constraint corresponds to a Dirichlet constraint
        dof_position_counter = 0
        for (d, c) in coeffs
            nested_eq = get(ch.dofmapping, d, 0)
            dof_position_counter += 1
            nested_eq == 0 && continue # skip as d is not in the prescribed dofs and therefore not nested
            
            nested_coeffs = ch.dofcoefficients[nested_eq]
            if !(nested_coeffs === nothing || isempty(nested_coeffs)) # nothing means Dirichlet, empty means Dirichlet but through AffineConstraint
                
                # add the dof to the affine_cdof_ordering
                _assign_new_index!(affine_cdof_ordering, d)
                # add the equation to affine_equation_ordering
                _assign_new_index!(affine_equation_ordering, nested_eq)

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
    _create_rhs_affine_constraint_matrix(ch::ConstraintHandler{DH,T}, affine_equation_ordering::Dict{Int,Int}) where {DH,T}

Create and return the constraint matrix, `C`, and the inhomogeneities, `g`, from the nested affine
constraints in `ch`. The constraint matrix relates constrained affine dofs, `a_c`, and free, `a_f`, degrees of freedom via
`A * a_c = C * a_f + g`. A mapping `affine_fdof_ordering` is also returned to index `C`, i.e. it maps "free" dof => column of C.
The rows are indexed using `affine_equation_ordering` from `_create_lhs_affine_constraint_matrix`.

!!! note
    In this case, the system `A * a_c = C * a_f + g` only contains the affine 
    constraints that are nested. Therefore this function is not designed to be used after the
    `ConstraintHandler` has been closed.
"""
function _create_rhs_affine_constraint_matrices(ch::ConstraintHandler{DH,T}, affine_equation_ordering::Dict{Int,Int}) where {DH,T}

    n_nested_constraints = length(affine_equation_ordering)
    I = Int[]; J = Int[]; V = T[]
    g = Vector{T}(undef, n_nested_constraints) # inhomogeneities

    # maps the free dofs to a position in `a_f`
    affine_fdof_ordering = Dict{Int,Int}()

    for (eq, coeffs) in enumerate(ch.dofcoefficients)
        (isnothing(coeffs) || !haskey(affine_equation_ordering, eq)) && continue
        i = affine_equation_ordering[eq]
        if isempty(coeffs) && haskey(affine_equation_ordering, eq)
            # the constraint was filled with nested dofs and now the dof coefficients are empty
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
    C = SparseArrays.sparse(I, J, V, n_nested_constraints, n)

    return C, g, affine_fdof_ordering
end

function _update_dof_coefficents!(dc::Vector{Union{Nothing,DofCoefficients{T}}}, C::AbstractMatrix, affine_equation_ordering::Dict{Int,Int}, affine_fdof_ordering::Dict{Int,Int}) where {T}
    
    affine_fdof_mapping⁻¹ = Dict(v => k for (k, v) in affine_fdof_ordering) # Bijections.jl could avoid this but not really worth it

    for (eq, i) in pairs(affine_equation_ordering)
        coeffs = Pair{Int,T}[]
        for j in axes(C, 2)
            C[i,j] == zero(T) && continue # skip zero entry
            d = affine_fdof_mapping⁻¹[j]
            push!(coeffs, (d => C[i,j]))
        end
        # overwrite the old coefficients
        dc[eq] = coeffs
    end
    return dc
end

"""
    isnested(ch::ConstraintHandler)

Check if the constraint handler has any nested affine dofs.
"""
function isnested(ch::ConstraintHandler)
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