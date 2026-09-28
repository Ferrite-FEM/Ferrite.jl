"""
    _untangle_affine_constraints!(ch::ConstraintHandler)

Untangle the affine constraints in `ch`, i.e. rewrite them such that no master dof of an
affine constraint is itself constrained by an affine constraint. For example, the system

    u1 = u2 + u5
    u2 = u3 + 4 * u10 + 4.0
    u9 = 3 * u2 - 2.0

is tangled since `u2` appears both as a master and a slave dof. After untangling it reads

    u1 = u3 + 4 * u10 + u5 + 4.0
    u2 = u3 + 4 * u10 + 4.0
    u9 = 3 * u3 + 12 * u10 + 10.0

Dirichlet-type master dofs (constraints without coefficients, e.g. `u3 = f3(t)`) are not
substituted here; their contribution to the inhomogeneities is computed in `update!`.

The constraints are viewed as a directed graph where constraint `i` points to constraint
`j` when the slave dof of `j` is a master dof of `i`. The strongly connected components of
this graph are computed with Tarjan's algorithm, which emits them in reverse topological
order. Processing the components in this order means that when a constraint is reached, all
constraints it depends on are already untangled and can be substituted directly, so the
cost is proportional to the total size of the substituted constraints. Only components with
a genuine cycle (e.g. `u1 = 2 * u2 + 1, u2 = u1 + 1`) require solving a (small) linear system.
"""
function _untangle_affine_constraints!(ch::ConstraintHandler{DH, Tv, Ti}) where {DH, Tv, Ti}
    @assert _istangled(ch) "ConstraintHandler is not tangled"
    # Which constraints are affine is decided once, from the original constraints: a
    # constraint that simplifies to a constant during untangling must still be substituted.
    isaffine = _isaffine(ch)
    sccs = _affine_constraint_sccs(ch, isaffine)
    resolved = falses(length(isaffine))
    spa = _SparseAccumulator{Tv, Ti}(ndofs(ch.dh))
    for scc in sccs
        if length(scc) == 1 && !_has_affine_master(ch, isaffine, scc[1])
            # no tangled master, leave the user given coefficients untouched
        elseif length(scc) == 1 && !_has_self_loop(ch, scc[1])
            _substitute_resolved!(ch, spa, isaffine, scc[1], resolved)
        else
            _untangle_cyclic_scc!(ch, spa, isaffine, scc, resolved)
        end
        for i in scc
            resolved[i] = true
        end
    end
    @assert !_istangled(ch)
    return ch
end

# Affine constraints are the ones with (non-empty) coefficients; empty ones act as Dirichlet
function _isaffine(ch::ConstraintHandler)
    return BitVector(c !== nothing && !isempty(c) for c in ch.dofcoefficients)
end

# Index of the affine constraint with slave dof `d`, or 0
function _affine_constraint_index(ch::ConstraintHandler{DH, Tv, Ti}, isaffine::BitVector, d) where {DH, Tv, Ti}
    # ch.isconstrained is only filled in close!, but when it is it avoids most dict lookups
    isempty(ch.isconstrained) || ch.isconstrained[d] || return zero(Ti)
    j = get(ch.dofmapping, d, zero(Ti))
    return (j != 0 && isaffine[j]) ? j : zero(Ti)
end

# Whether constraint `i` has a master dof constrained by an affine constraint
function _has_affine_master(ch::ConstraintHandler, isaffine::BitVector, i)
    return any(dc -> _affine_constraint_index(ch, isaffine, dc.first) != 0, ch.dofcoefficients[i]::DofCoefficients)
end

function _has_self_loop(ch::ConstraintHandler, i)
    d = ch.prescribed_dofs[i]
    return any(dc -> dc.first == d, ch.dofcoefficients[i])
end

"""
    _istangled(ch::ConstraintHandler)

Check if the constraint handler has any tangled dofs. An example of a tangled dof is

    u1 = u2 + u5
    u2 = u3 + 4 * u10 + 4.0.

Here, `u2` is a tangled dof as it appears on the left- and right-hand side of the constraints.
"""
function _istangled(ch::ConstraintHandler)
    isaffine = _isaffine(ch)
    for (i, coeffs) in enumerate(ch.dofcoefficients)
        isaffine[i] || continue
        for (d, _) in coeffs
            _affine_constraint_index(ch, isaffine, d) != 0 && return true
        end
    end
    return false
end

# Strongly connected components of the affine constraint graph in reverse topological order
# (every component is emitted after all components it depends on). Iterative Tarjan.
function _affine_constraint_sccs(ch::ConstraintHandler, isaffine::BitVector)
    n = length(isaffine)
    index = zeros(Int, n)     # 0: not visited
    lowlink = zeros(Int, n)
    onstack = falses(n)
    stack = Int[]
    sccs = Vector{Int}[]
    # DFS state: (node, position in its coefficient list)
    dfs = Tuple{Int, Int}[]
    counter = 0
    for root in 1:n
        (index[root] != 0 || !isaffine[root]) && continue
        counter += 1
        index[root] = lowlink[root] = counter
        push!(stack, root); onstack[root] = true
        push!(dfs, (root, 1))
        while !isempty(dfs)
            v, pos = dfs[end]
            coeffs = ch.dofcoefficients[v]::DofCoefficients
            descended = false
            while pos <= length(coeffs)
                w = _affine_constraint_index(ch, isaffine, coeffs[pos].first)
                pos += 1
                w == 0 && continue
                if index[w] == 0
                    dfs[end] = (v, pos)
                    counter += 1
                    index[w] = lowlink[w] = counter
                    push!(stack, w); onstack[w] = true
                    push!(dfs, (w, 1))
                    descended = true
                    break
                elseif onstack[w]
                    lowlink[v] = min(lowlink[v], index[w])
                end
            end
            descended && continue
            # all successors of v visited
            pop!(dfs)
            if lowlink[v] == index[v]
                scc = Int[]
                while true
                    w = pop!(stack); onstack[w] = false
                    push!(scc, w)
                    w == v && break
                end
                push!(sccs, scc)
            end
            if !isempty(dfs)
                u = dfs[end][1]
                lowlink[u] = min(lowlink[u], lowlink[v])
            end
        end
    end
    return sccs
end

# Sparse accumulator: dense value array indexed by dof plus the list of touched dofs.
struct _SparseAccumulator{Tv, Ti}
    values::Vector{Tv}
    touched::Vector{Ti}
    istouched::BitVector
end
function _SparseAccumulator{Tv, Ti}(n::Int) where {Tv, Ti}
    return _SparseAccumulator{Tv, Ti}(zeros(Tv, n), Ti[], falses(n))
end
function _spa_add!(spa::_SparseAccumulator, d, v)
    if !spa.istouched[d]
        spa.istouched[d] = true
        push!(spa.touched, d)
    end
    spa.values[d] += v
    return spa
end
# Move the accumulated (nonzero) entries into `coeffs`, sorted by dof, and reset
function _spa_collect!(coeffs::DofCoefficients, spa::_SparseAccumulator)
    sort!(spa.touched)
    for d in spa.touched
        v = spa.values[d]
        spa.values[d] = zero(v)
        spa.istouched[d] = false
        iszero(v) && continue
        push!(coeffs, d => v)
    end
    empty!(spa.touched)
    return coeffs
end

# Expand the constraint `i` into the accumulator by substituting the already resolved affine
# masters. Masters that are affine but not resolved (i.e. in the same strongly connected
# component) are pushed to `cyclic` instead. Returns the accumulated inhomogeneity.
function _expand_constraint!(spa::_SparseAccumulator, cyclic, ch::ConstraintHandler{DH, Tv}, isaffine, i, resolved) where {DH, Tv}
    coeffs = ch.dofcoefficients[i]::DofCoefficients
    b = ch.affine_inhomogeneities[i]::Tv
    for (d, c) in coeffs
        j = _affine_constraint_index(ch, isaffine, d)
        if j == 0
            _spa_add!(spa, d, c)
        elseif resolved[j]
            for (dj, cj) in ch.dofcoefficients[j]::DofCoefficients
                _spa_add!(spa, dj, c * cj)
            end
            b += c * ch.affine_inhomogeneities[j]::Tv
        else
            push!(cyclic, j => c)
        end
    end
    return b
end

# The coefficient vectors in `ch` may be shared with the user's `AffineConstraint`s (and
# between constraints), so they are never modified in place; new vectors are stored instead.
function _set_untangled!(ch::ConstraintHandler, i, coeffs::DofCoefficients, b)
    ch.dofcoefficients[i] = coeffs
    ch.affine_inhomogeneities[i] = b
    ch.inhomogeneities[i] = b # effective inhomogeneity, recomputed in update!
    return ch
end

function _substitute_resolved!(ch::ConstraintHandler{DH, Tv, Ti}, spa, isaffine, i, resolved) where {DH, Tv, Ti}
    cyclic = Pair{Int, Tv}[]
    b = _expand_constraint!(spa, cyclic, ch, isaffine, i, resolved)
    @assert isempty(cyclic)
    return _set_untangled!(ch, i, _spa_collect!(DofCoefficients{Tv, Ti}(), spa), b)
end

# Untangle a component whose constraints depend on each other cyclically by solving the
# linear system `A * u_c = C * u_f + g` for the slave dofs `u_c` of the component, where
# `u_f` are the (already resolved) master dofs.
function _untangle_cyclic_scc!(ch::ConstraintHandler{DH, Tv, Ti}, spa, isaffine, scc::Vector{Int}, resolved) where {DH, Tv, Ti}
    k = length(scc)
    local_index = Dict{Int, Int}(i => li for (li, i) in enumerate(scc))
    # Expand all constraints of the component; collect the union of master dofs
    expanded = Vector{DofCoefficients{Tv, Ti}}(undef, k)
    cyclic = [Pair{Int, Tv}[] for _ in 1:k]
    g = Vector{Tv}(undef, k)
    for (li, i) in enumerate(scc)
        g[li] = _expand_constraint!(spa, cyclic[li], ch, isaffine, i, resolved)
        expanded[li] = _spa_collect!(DofCoefficients{Tv, Ti}(), spa)
    end
    masters = unique!(sort!(Ti[d for e in expanded for (d, _) in e]))
    master_index = Dict{Ti, Int}(d => c for (c, d) in enumerate(masters))
    m = length(masters)
    # Build A (k × k) and the right hand side [C g] (k × (m + 1)). Every slave in the
    # component depends on (almost) every master of the component, so the result is dense
    # and a dense solve is appropriate.
    A = Matrix{Tv}(LinearAlgebra.I, k, k)
    B = zeros(Tv, k, m + 1)
    for li in 1:k
        for (j, c) in cyclic[li]
            A[li, local_index[j]] -= c
        end
        for (d, c) in expanded[li]
            B[li, master_index[d]] += c
        end
        B[li, m + 1] = g[li]
    end
    F = LinearAlgebra.lu(A; check = false)
    if !LinearAlgebra.issuccess(F)
        throw(
            ArgumentError(
                "the affine constraints contain a cycle that cannot be resolved, e.g. due to " *
                    "redundant constraints such as u1 = u2 and u2 = u1"
            )
        )
    end
    X = F \ B
    for (li, i) in enumerate(scc)
        coeffs = DofCoefficients{Tv, Ti}()
        for (c, d) in enumerate(masters)
            v = X[li, c]
            iszero(v) && continue
            push!(coeffs, d => v)
        end
        _set_untangled!(ch, i, coeffs, X[li, m + 1])
    end
    return ch
end
