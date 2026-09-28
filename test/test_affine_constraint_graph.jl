using Test, LinearAlgebra, Random
import Graphs
using Ferrite: _strongly_connected_components, _untangle_affine_constraints, _solve_affine_component

@testset "strongly connected components" begin
    # Compare with Graphs.jl's implementation of the same algorithm. Component and
    # vertex order can differ, but the partition and dependency ordering must agree.
    @testset "Graphs.jl comparison" begin
        cases = [
            Vector{Int}[], [Int[], Int[]], [[1]],
            [[2], [3, 3], [2], [1], [5]],
            [[2], [3], Int[]], [[2], [3], [1]],
            [[2], [1], [4], [3], Int[]],
            vcat([collect(2:32)], [Int[] for _ in 2:32]),
        ]
        for adjacency in cases
            graph = Graphs.SimpleDiGraph(length(adjacency))
            for (i, neighbors) in enumerate(adjacency), j in neighbors
                Graphs.add_edge!(graph, i, j)
            end
            ours = _strongly_connected_components(adjacency)
            reference = Graphs.strongly_connected_components_tarjan(graph)
            @test Set(Set.(ours)) == Set(Set.(reference))
            for components in (ours, reference)
                order = Dict(v => i for (i, component) in enumerate(components) for v in component)
                @test all(order[j] <= order[i] for i in eachindex(adjacency) for j in adjacency[i])
            end
        end
    end

    @test isempty(_strongly_connected_components(Vector{Int}[]))
    @test Set(Set.(_strongly_connected_components([Int[], Int[]]))) == Set([Set([1]), Set([2])])
    @test _strongly_connected_components([[1]]) == [[1]]
    # Each list contains the outgoing neighbors of its vertex:
    #
    #                 +-------+
    #                 |       v
    #   4 ---> 1 ---> 2 -----> 3       5 --+
    #                 ^       |       ^  |
    #                 +-------+       +--+
    #
    # There are two edges from 2 to 3, one back from 3 to 2, and a self-loop at 5.
    graph = [[2], [3, 3], [2], [1], [5]]
    original = deepcopy(graph)
    components = _strongly_connected_components(graph)
    @test Set(Set.(components)) == Set([Set([1]), Set([2, 3]), Set([4]), Set([5])])
    order = Dict(v => i for (i, component) in enumerate(components) for v in component)
    @test order[2] == order[3] < order[1] < order[4]
    @test graph == original

    # Compare the partition with mutual reachability, independently of Tarjan's
    # traversal order, including edges to vertices whose SCC was already completed.
    rng = MersenneTwister(42)
    for _ in 1:40
        n = 12
        reachable = rand(rng, n, n) .< 0.1
        graph = [findall(reachable[i, :]) for i in 1:n]
        for i in 1:n
            reachable[i, i] = true
        end
        for k in 1:n, i in 1:n, j in 1:n
            reachable[i, j] |= reachable[i, k] && reachable[k, j]
        end
        components = _strongly_connected_components(graph)
        @test sort(vcat(components...)) == collect(1:n)
        order = Dict(v => i for (i, component) in enumerate(components) for v in component)
        @test all((order[i] == order[j]) == (reachable[i, j] && reachable[j, i]) for i in 1:n, j in 1:n)
        @test all(order[j] <= order[i] for i in 1:n for j in graph[i])
    end

    # Deep traversal uses explicit stacks rather than recursive calls.
    n = 20_000
    chain = [i == n ? Int[] : [i + 1] for i in 1:n]
    @test _strongly_connected_components(chain) == [[i] for i in n:-1:1]
end

@testset "affine substitution" begin
    # The documented example, with no mesh or ConstraintHandler.
    coefficients = [[2 => 1.0, 4 => 1.0], [3 => 2.0]]
    constants = [0.0, 1.0]
    original = deepcopy(coefficients)
    result, values = _untangle_affine_constraints([1, 2], coefficients, constants)
    @test result == [[3 => 2.0, 4 => 1.0], [3 => 2.0]]
    @test values == [1.0, 1.0]
    @test coefficients == original
    @test constants == [0.0, 1.0]

    # A cycle feeds two successive substitutions; equation order and variable order
    # deliberately differ. u2 = 2u3 + 1, u3 = u2 + u6 + 1; u1 = u2 + u5; u4 = 3u1 - 2.
    result, values = _untangle_affine_constraints(
        [4, 3, 1, 2],
        [[1 => 3.0], [2 => 1.0, 6 => 1.0], [2 => 1.0, 5 => 1.0], [3 => 2.0]],
        [-2.0, 1.0, 0.0, 1.0],
    )
    @test result == [[5 => 3.0, 6 => -6.0], [6 => -1.0], [5 => 1.0, 6 => -2.0], [6 => -2.0]]
    @test values == [-11.0, -2.0, -3.0, -3.0]

    # Cancellation creates a constant that must still be substituted into its user.
    result, values = _untangle_affine_constraints(
        [1, 2, 3, 4],
        [[2 => 1.0], [3 => 1.0, 4 => -2.0], [7 => 1.0], [7 => 0.5]],
        [0.0, 0.0, 10.0, 2.0],
    )
    @test all(isempty, result[1:2])
    @test values[1:2] == [6.0, 6.0]
    # Explicitly constant equations are also supported by the standalone interface.
    result, values = _untangle_affine_constraints([2, 1], [Pair{Int, Float64}[], [2 => 2.0]], [3.0, 1.0])
    @test all(isempty, result)
    @test values == [3.0, 7.0]

    shared = [3 => 1.0]
    result, values = _untangle_affine_constraints([1, 2, 3], [shared, shared, [4 => 2.0]], [0.0, 1.0, 2.0])
    @test shared == [3 => 1.0]
    @test result == fill([4 => 2.0], 3)
    @test values == [2.0, 3.0, 2.0]
    result, values = _untangle_affine_constraints([1, 2], [[2 => 1.0, 5 => 1.0, 2 => 3.0], [5 => -0.25, 6 => 1.0]], [0.0, 0.0])
    @test result[1] == [6 => 4.0]

    result, values = _untangle_affine_constraints(Int[], Vector{Pair{Int, Float64}}[], Float64[])
    @test isempty(result) && isempty(values)
    @test_throws DimensionMismatch _untangle_affine_constraints([1], [[2 => 1.0]], Float64[])
    @test_throws ArgumentError _untangle_affine_constraints([1, 1], [[2 => 1.0], [3 => 1.0]], [0.0, 0.0])

    # Compare complete systems with a direct dense solve. A strict contraction gives
    # nonsingular systems, including cycles, disconnected graphs, and constant rows.
    rng = MersenneTwister(28)
    for _ in 1:40
        n, m = 12, 4
        slaves = randperm(rng, n)
        coefficients = [[rand(rng, 1:(n + m)) => (rand(rng) - 0.5) * 0.1 for _ in 1:rand(rng, 0:5)] for _ in 1:n]
        constants = randn(rng, n)
        result, values = _untangle_affine_constraints(slaves, coefficients, constants)
        external = randn(rng, m)
        A = Matrix{Float64}(I, n, n)
        rhs = zeros(n)
        for (i, slave) in enumerate(slaves)
            rhs[slave] = constants[i]
            for (d, c) in coefficients[i]
                if d <= n
                    A[slave, d] -= c
                else
                    rhs[slave] += c * external[d - n]
                end
            end
        end
        @test all(d > n for row in result for (d, _) in row)
        actual = zeros(n)
        for (i, slave) in enumerate(slaves)
            actual[slave] = values[i] + sum((c * external[d - n] for (d, c) in result[i]); init = 0.0)
        end
        @test actual ≈ A \ rhs
    end
end

@testset "standalone cyclic component solve" begin
    # Equation identifiers are distinct from row numbers and external variable IDs.
    cyclic = [[20 => 2.0], [10 => 1.0]]
    expanded = [Pair{Int, Float64}[], [7 => 1.0]]
    constants = [1.0, 1.0]
    inputs = deepcopy((cyclic, expanded, constants))
    result, values = _solve_affine_component([10, 20], cyclic, expanded, constants)
    @test result == [[7 => -2.0], [7 => -1.0]]
    @test values == [-3.0, -2.0]
    @test (cyclic, expanded, constants) == inputs

    # Self-loop and repeated coefficients in both the cycle and external terms.
    result, values = _solve_affine_component([10], [[10 => 0.25, 10 => 0.25]], [[7 => 1.0, 7 => 2.0]], [1.0])
    @test result == [[7 => 6.0]]
    @test values == [2.0]
    @test_throws ArgumentError _solve_affine_component([1, 2], [[2 => 1.0], [1 => 1.0]], [Pair{Int, Float64}[], Pair{Int, Float64}[]], [0.0, 0.0])
    @test_throws DimensionMismatch _solve_affine_component([1], [[1 => 0.5]], [Pair{Int, Float64}[]], Float64[])

    # Dense/sparse boundary and the generic dense path for BigFloat. All ring
    # equations resolve to u_i = 2u_master + 2 (or u_i = 2 without a master).
    for T in (Float32, Float64, ComplexF64, BigFloat), n in (1, 64, 65), with_master in (false, true)
        cyclic = [[mod1(i + 1, n) => T(0.5)] for i in 1:n]
        expanded = [with_master ? [Int32(n + 1) => one(T)] : Pair{Int32, T}[] for _ in 1:n]
        result, values = _solve_affine_component(collect(1:n), cyclic, expanded, ones(T, n))
        @test result isa Vector{Vector{Pair{Int32, T}}}
        @test values isa Vector{T}
        @test values ≈ fill(T(2), n)
        if with_master
            @test all(row -> first.(row) == [Int32(n + 1)] && last.(row) ≈ [T(2)], result)
        else
            @test all(isempty, result)
        end
    end
    n = 65
    @test_throws ArgumentError _solve_affine_component(collect(1:n), [[mod1(i + 1, n) => 1.0] for i in 1:n], [Pair{Int, Float64}[] for _ in 1:n], zeros(n))
end
