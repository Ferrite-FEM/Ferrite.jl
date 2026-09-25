"""
    _untangle_affine_constraints!(ch::ConstraintHandler)

Untangle the affine constraints in `ch`. This is best illustrated using an example. The following system has
tangled constraints as `u2` appears as a master and a slave dof.

    u1 = u2 + u5
    u2 = u3 + 4 * u10 + 4.0
    u9 = 3 * u2 - 2.0

To untangle this the following linear system is assembled, here `a_c` and `a_f` are the vectors of the individual dofs `u_i`.

    A * a_c = C * a_f + g.

Concretely, for the above example we get

    | 1  -1  ⋅ | |u1|   |1  ⋅||u5|   | 1.0|
    | ⋅   1  ⋅ | |u2| = |⋅  1||u3| + | 4.0|
    | ⋅  -3  1 | |u9|   |⋅  ⋅|       |-2.0|.
    
Solving this system we find the new master/slave dofs and their coefficients

    |u1|   |1  1||u5|   | 5.0|
    |u2| = |0  1||u3| + | 4.0|
    |u9|   |0  3|       |10.0|

which are then used to update the `ConstraintHandler` accordingly. A couple of things to note here:

    * If a Dirichlet-type dof appears in a tangled fashion, i.e. `u3 = f3(t)` in the system above, this is not included in `A` or `C`
    as Ferrite allows tangled dofs if they of Dirichlet-type.

    * The case that a Dirichlet dof is also a master dof i.e., `u9 = f9(t)` in the above system, does not occur as Ferrite overwrites master
    dofs in `add!` to ensure no two constraints share a master dof.

!!! warning
    As the system `A * a_c = C * a_f + g` only contains the affine
    constraints that are tangled. Therefore, this function is not designed to be called when the
    constraints are not tangled.

"""
function _untangle_affine_constraints!(ch::ConstraintHandler)
    @assert istangled(ch) "ConstraintHandler is not tangled"
    A, affine_equation_ordering, new_dofcoefficients = _create_lhs_affine_constraint_matrix(ch)
    C, g, affine_fdof_ordering = _create_rhs_affine_constraint_matrices(ch, new_dofcoefficients, affine_equation_ordering)

    luA = try
        LinearAlgebra.lu(A; check = true)
    catch e
        if e isa LinearAlgebra.SingularException
            throw(
                ArgumentError(
                    "the affine constraints are tangled and untangling them fails. " *
                        "This can be due to e.g. redundant constraints. A possibility to avoid this is to guarantee that " *
                        "the constraints are not tangled before calling close!"
                )
            )
        else
            rethrow(e)
        end
    end

    A⁻¹C = _sparse_column_wise_solve(luA, C)
    _update_dof_coefficients!(new_dofcoefficients, A⁻¹C, affine_equation_ordering, affine_fdof_ordering)

    ldiv!(luA, g)

    # we need to update ch.affine_inhomogeneities NOT ch.inhomogeneities
    # as ch.inhomogeneities will be computed in update!
    for (k, v) in affine_equation_ordering
        ch.affine_inhomogeneities[k] = g[v]
    end

    # finally update the dofcoefficients in the constraint handler
    ch.dofcoefficients .= new_dofcoefficients

    @assert !istangled(ch)
    return ch
end

"""
    _create_lhs_affine_constraint_matrix(ch::ConstraintHandler{DH, T}) where {DH, T}

Create and returns the left-hand side constraint matrix `A` from the system `A * a_c = C * a_f + g`. As `A` only contains the
tangled affine constraints its structure is built from the ground up. This means there is a mapping required to associate
each row in `A` to its original position. To do this `affine_equation_ordering` returned. Finally, `new_dofcoefficients` are returned
which have the entries that are now in `A` removed.

"""
function _create_lhs_affine_constraint_matrix(ch::ConstraintHandler{DH, Tv, Ti}) where {DH, Tv, Ti}

    # maps the constrained dofs to a position in `a_c`
    affine_cdof_ordering = Dict{Int, Int}()
    # maps the constraint equation to a row in `A * a_c = C * a_f + g`
    affine_equation_ordering = Dict{Int, Int}()
    # collect the position of the dof coefficients that need to be removed for ch.dofcoefficients
    dofcoeffs_to_remove = Dict{Int, Vector{Int}}()

    I = Ti[]; J = Ti[]; V = Tv[]
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
    @assert m == n "The matrix A has dimensions m = $m != n = $n"

    A = SparseArrays.sparse(I, J, V, m, n)

    # finally remove the entries that have been moved into A so that new_dofcoefficients can be used to construct `C`
    new_dofcoefficients = deepcopy(ch.dofcoefficients)
    for (k, v) in dofcoeffs_to_remove
        deleteat!(new_dofcoefficients[k], v)
    end

    return A, affine_equation_ordering, new_dofcoefficients
end

"""
    _create_rhs_affine_constraint_matrices(ch::ConstraintHandler{DH, Tv, Ti}, new_dofcoefficients, affine_equation_ordering::Dict{Int, Int}) where {DH, Tv, Ti}

Create and returns the right-hand side constraint matrix `C` and its inhomogenties `g` from the system `A * a_c = C * a_f + g`. Returned are `C`,
`g` and `affine_fdof_ordering` which maps the "free" dofs to the columns of `C`.

"""
function _create_rhs_affine_constraint_matrices(ch::ConstraintHandler{DH, Tv, Ti}, new_dofcoefficients::Vector{Union{Nothing, DofCoefficients{Tv, Ti}}}, affine_equation_ordering::Dict{Int, Int}) where {DH, Tv, Ti}

    n_tangled_constraints = length(affine_equation_ordering)
    I = Ti[]; J = Ti[]; V = Tv[]
    g = Vector{Tv}(undef, n_tangled_constraints) # inhomogeneities

    # maps the free dofs to a position in `a_f`
    affine_fdof_ordering = Dict{Int, Int}()

    for (eq, coeffs) in enumerate(new_dofcoefficients)
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

"""
    _update_dof_coefficients!(dc::Vector{Union{Nothing, DofCoefficients{Tv, Ti}}}, A⁻¹C::AbstractMatrix, affine_equation_ordering::Dict{Int, Int}, affine_fdof_ordering::Dict{Int, Int}) where {Tv, Ti}

Update the dof coefficients `dc` using the constraint matrix `A⁻¹C` and the mappings `affine_equation_ordering` and `affine_fdof_ordering`.
"""
function _update_dof_coefficients!(dc::Vector{Union{Nothing, DofCoefficients{Tv, Ti}}}, A⁻¹C::SparseMatrixCSC, affine_equation_ordering::Dict{Int, Int}, affine_fdof_ordering::Dict{Int, Int}) where {Tv, Ti}

    affine_fdof_mapping⁻¹ = Dict(v => k for (k, v) in affine_fdof_ordering) # Bijections.jl could avoid this but probably not worth it
    affine_equation_ordering⁻¹ = Dict(v => k for (k, v) in affine_equation_ordering)

    for (k, _) in affine_equation_ordering
        dc[k] = DofCoefficients{Tv, Ti}()
    end

    for j in axes(A⁻¹C, 2)
        for nz_i in nzrange(A⁻¹C, j)
            i = A⁻¹C.rowval[nz_i]
            dof = affine_fdof_mapping⁻¹[j]
            coeffs = dc[affine_equation_ordering⁻¹[i]]
            push!(coeffs, (dof => A⁻¹C.nzval[nz_i]))
        end
    end
    return dc
end

"""
    istangled(ch::ConstraintHandler)

Check if the constraint handler has any tangled dofs. An example of a tangled dof is

    u1 = u2 + u5
    u2 = u3 + 4 * u10 + 4.0.

Here, `u2` is a tangled dof as it appears on the left- and right-hand side of the constraints.
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

function _assign_new_index!(d::Dict{Int, Int}, key::Int)
    return get!(d, key, length(d) + 1)
end

"""
    _sparse_column_wise_solve(A::SparseArrays.UMFPACK.UmfpackLU{T, TiA}, C::SparseMatrixCSC{T, TiC}) where {T, TiA, TiC}

Perform a column wise solve of `AX = C` where `X` is expected to be a sparse matrix, this avoids the dense construction of `X`.
The LU decomposition of `A` should be passed
"""
function _sparse_column_wise_solve(A::SparseArrays.UMFPACK.UmfpackLU{T, TiA}, C::SparseMatrixCSC{T, TiC}) where {T, TiA, TiC}
    (m, n) = size(C)
    I = TiC[]; J = TiC[]; V = T[]
    sh = SparseArrays.nnz(C)
    sizehint!(I, sh); sizehint!(J, sh); sizehint!(V, sh)
    lhs = zeros(T, m)
    rhs = zeros(T, m)
    for j in axes(C, 2)
        iszero(C[:, j]) && continue
        copy!(rhs, C[:, j])
        ldiv!(lhs, A, rhs)
        for (i, v) in pairs(lhs)
            v == zero(T) && continue
            push!(I, i)
            push!(J, j)
            push!(V, v)
        end
    end
    return SparseArrays.sparse(I, J, V, m, n)
end
