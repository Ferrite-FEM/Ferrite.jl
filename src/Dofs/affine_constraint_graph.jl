# This is a direct implementation of the standard Tarjan SCC algorithm, also provided
# by Graphs.jl. Keep it locally until Graphs.jl's traversal is fast enough; then replace
# it with the call below. See https://github.com/JuliaGraphs/Graphs.jl/pull/527.

# The read-only adapter avoids copying our adjacency lists into a SimpleDiGraph and
# implements only the graph operations needed by Tarjan. Enable it when switching:
# import Graphs
# struct _ConstraintGraph{A} <: Graphs.AbstractGraph{Int}
#     adjacency::A
# end
# Graphs.nv(g::_ConstraintGraph) = length(g.adjacency)
# Graphs.vertices(g::_ConstraintGraph) = Base.OneTo(Graphs.nv(g))
# Graphs.outneighbors(g::_ConstraintGraph, v::Integer) = g.adjacency[v]
# Graphs.is_directed(::Type{<:_ConstraintGraph}) = true

"""
    _strongly_connected_components(adjacency)

Return strongly connected components of a directed graph as a `Vector{Vector{Int}}`.
`adjacency[i]` lists the vertices that vertex `i` points to; vertices are numbered
`1:length(adjacency)`. Self-loops, repeated edges, and isolated vertices are allowed.
Components are emitted in reverse topological order: for an edge `i → j` between
components, the component containing `j` comes first. Ordering within a component,
and between unrelated components, is not part of the interface. The input is unchanged.

For example, `[ [2], [3], [2], [1] ]` describes `1 → 2 ↔ 3` and `4 → 1`.
The returned components are `[ [3, 2], [1], [4] ]`: resolve the cycle before its users.
An iterative Tarjan traversal avoids recursion on long chains and takes O(V + E) work.
"""
function _strongly_connected_components(adjacency::AbstractVector{<:AbstractVector{<:Integer}})
    Base.require_one_based_indexing(adjacency)
    foreach(Base.require_one_based_indexing, adjacency)
    # return Graphs.strongly_connected_components_tarjan(_ConstraintGraph(adjacency))
    n = length(adjacency)
    index = zeros(Int, n)     # 0: not visited
    lowlink = zeros(Int, n)
    onstack = falses(n)
    stack = Int[]
    sccs = Vector{Int}[]
    # DFS state: (node, position in its adjacency list)
    dfs = Tuple{Int, Int}[]
    counter = 0
    for root in 1:n
        index[root] != 0 && continue
        counter += 1
        index[root] = lowlink[root] = counter
        push!(stack, root); onstack[root] = true
        push!(dfs, (root, 1))
        while !isempty(dfs)
            v, pos = dfs[end]
            neighbors = adjacency[v]
            descended = false
            while pos <= length(neighbors)
                w = neighbors[pos]
                pos += 1
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

"""
    _untangle_affine_constraints(slaves, coefficients, constants; nvariables)

Resolve equations `u[slaves[i]] = sum(c * u[d] for (d, c) in coefficients[i]) + constants[i]`.
`slaves` contains unique positive integer variable indices; `coefficients` is a vector
of vectors of `Pair{Ti, Tv}`; `constants` is a vector of the same value type `Tv`.
All arrays use one-based indexing. `nvariables` defaults to the largest variable index.

Return `(new_coefficients, new_constants)` in the original equation order. No returned
coefficient refers to a variable in `slaves`. Variables absent from `slaves` remain
symbolic, so callers can supply their values later. An empty coefficient list denotes a
constant equation and is substituted too. A singular cycle throws `ArgumentError`.

Inputs are not modified. The returned outer arrays are new, but untouched coefficient
vectors may be shared with the input. Rewritten vectors combine duplicate masters,
drop exact zeros, and sort by variable index; untouched vectors retain their ordering.
Storage types `Ti` and `Tv` are preserved (solutions must be representable in `Tv`).

For example, `slaves = [1, 2]`, `coefficients = [[2 => 1.0, 4 => 1.0], [3 => 2.0]]`,
and `constants = [0.0, 1.0]` describe `u1 = u2 + u4`, `u2 = 2u3 + 1`.
The result is `([[3 => 2.0, 4 => 1.0], [3 => 2.0]], [1.0, 1.0])`.
"""
function _untangle_affine_constraints(
        slaves::AbstractVector{<:Integer},
        coefficients::AbstractVector{<:AbstractVector{Pair{Ti, Tv}}},
        constants::AbstractVector{Tv};
        nvariables::Integer = max(
            maximum(slaves; init = 0),
            maximum((d for row in coefficients for (d, _) in row); init = 0),
        ),
    ) where {Ti <: Integer, Tv}
    Base.require_one_based_indexing(slaves, coefficients, constants)
    length(slaves) == length(coefficients) == length(constants) ||
        throw(DimensionMismatch("expected one coefficient vector and constant per slave"))
    equation_index = zeros(Int, nvariables)
    for (i, d) in enumerate(slaves)
        equation_index[d] == 0 || throw(ArgumentError("slave variable $d occurs more than once"))
        equation_index[d] = i
    end
    adjacency = [Int[equation_index[d] for (d, _) in row if equation_index[d] != 0] for row in coefficients]
    sccs = _strongly_connected_components(adjacency)
    # Replace outer entries, never mutate the user's coefficient vectors. Keep the
    # original equation_index even when substitution simplifies an equation to a constant.
    result = Vector{Pair{Ti, Tv}}[row isa Vector{Pair{Ti, Tv}} ? row : collect(row) for row in coefficients]
    values = collect(constants)
    resolved = falses(length(slaves))
    spa = _SparseAccumulator{Tv, Ti}(Int(nvariables))
    for scc in sccs
        i = first(scc)
        if length(scc) == 1 && isempty(adjacency[i])
            # No dependent equation: leave the user-provided coefficients untouched.
        elseif length(scc) == 1 && !(i in adjacency[i])
            cyclic = Pair{Int, Tv}[]
            values[i] = _expand_constraint!(spa, cyclic, result, values, equation_index, i, resolved)
            @assert isempty(cyclic)
            result[i] = _spa_collect!(Pair{Ti, Tv}[], spa)
        else
            expanded = Vector{Pair{Ti, Tv}}[]
            cyclic = [Pair{Int, Tv}[] for _ in scc]
            g = Tv[]
            for (li, j) in enumerate(scc)
                push!(g, _expand_constraint!(spa, cyclic[li], result, values, equation_index, j, resolved))
                push!(expanded, _spa_collect!(Pair{Ti, Tv}[], spa))
            end
            component_coefficients, component_constants = _solve_affine_component(scc, cyclic, expanded, g)
            for (li, j) in enumerate(scc)
                result[j] = component_coefficients[li]
                values[j] = component_constants[li]
            end
        end
        resolved[scc] .= true
    end
    return result, values
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
function _spa_collect!(coeffs::Vector{<:Pair}, spa::_SparseAccumulator)
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
function _expand_constraint!(spa::_SparseAccumulator, cyclic, coefficients, constants, equation_index, i, resolved)
    coeffs = coefficients[i]
    b = constants[i]
    for (d, c) in coeffs
        j = equation_index[d]
        if j == 0
            _spa_add!(spa, d, c)
        elseif resolved[j]
            for (dj, cj) in coefficients[j]
                _spa_add!(spa, dj, c * cj)
            end
            b += c * constants[j]
        else
            push!(cyclic, j => c)
        end
    end
    return b
end

"""
    _solve_affine_component(component, cyclic, expanded, constants)

Solve one cyclic component after its external dependencies have been substituted.
`component` lists the equation identifiers in row order. `cyclic[i]` contains
`equation_id => coefficient` pairs whose identifiers occur in `component`;
`expanded[i]` contains `variable_id => coefficient` pairs for external variables.
These two index spaces are separate. `constants[i]` is the constant term of row `i`.

Return `(new_coefficients, new_constants)` in component row order, using only external
variable indices. Inputs are unchanged; output coefficient vectors are sorted and
contain no duplicates or exact zeros. A singular system throws `ArgumentError`.

For example, `component = [10, 20]`, `cyclic = [[20 => 2.0], [10 => 1.0]]`,
`expanded = [Pair{Int, Float64}[], [7 => 1.0]]`, and `constants = [1.0, 1.0]`
represent `x10 = 2x20 + 1`, `x20 = x10 + u7 + 1`. The result is
`([[7 => -2.0], [7 => -1.0]], [-3.0, -2.0])`.
"""
function _solve_affine_component(
        component::AbstractVector{<:Integer},
        cyclic::AbstractVector{<:AbstractVector{<:Pair}},
        expanded::AbstractVector{<:AbstractVector{Pair{Ti, Tv}}},
        constants::AbstractVector{Tv},
    ) where {Ti, Tv}
    Base.require_one_based_indexing(component, cyclic, expanded, constants)
    k = length(component)
    k == length(cyclic) == length(expanded) == length(constants) ||
        throw(DimensionMismatch("expected one cyclic row, expanded row, and constant per equation"))
    allunique(component) || throw(ArgumentError("component equation identifiers must be unique"))
    local_index = Dict(i => li for (li, i) in enumerate(component))
    masters = unique!(sort!(Ti[d for e in expanded for (d, _) in e]))
    master_index = Dict{Ti, Int}(d => c for (c, d) in enumerate(masters))
    m = length(masters)
    # Build A (k × k) and the right hand side [C constants] (k × (m + 1)). Although the
    # solution can be dense, A can remain sparse even for a large component (e.g. a
    # ring of constraints). Keep small components and types unsupported by sparse LU
    # on the dense path, but avoid quadratic storage for large sparse cycles.
    if k <= 64 || !(Tv <: Union{Float32, Float64, ComplexF32, ComplexF64})
        A = Matrix{Tv}(LinearAlgebra.I, k, k)
        for li in 1:k, (j, c) in cyclic[li]
            A[li, local_index[j]] -= c
        end
    else
        rows = collect(1:k)
        cols = collect(1:k)
        values = ones(Tv, k)
        for li in 1:k, (j, c) in cyclic[li]
            push!(rows, li)
            push!(cols, local_index[j])
            push!(values, -c)
        end
        A = SparseArrays.sparse(rows, cols, values, k, k)
    end
    B = zeros(Tv, k, m + 1)
    for li in 1:k
        for (d, c) in expanded[li]
            B[li, master_index[d]] += c
        end
        B[li, m + 1] = constants[li]
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
    new_coefficients = Vector{Pair{Ti, Tv}}[]
    new_constants = Tv[]
    for li in 1:k
        coeffs = Pair{Ti, Tv}[]
        for (c, d) in enumerate(masters)
            v = X[li, c]
            iszero(v) && continue
            push!(coeffs, d => v)
        end
        push!(new_coefficients, coeffs)
        push!(new_constants, X[li, m + 1])
    end
    return new_coefficients, new_constants
end
