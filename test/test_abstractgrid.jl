@testset "AbstractGrid" begin

    struct SmallGrid{dim,N,C<:Ferrite.AbstractCell} <: Ferrite.AbstractGrid{dim}
        nodes_test::Vector{NTuple{dim,Float64}}
        cells_test::NTuple{N,C}
    end

    Ferrite.getcells(grid::SmallGrid) = grid.cells_test
    Ferrite.getcells(grid::SmallGrid, v::Union{Int, Vector{Int}}) = grid.cells_test[v]
    Ferrite.getncells(grid::SmallGrid{dim,N}) where {dim,N} = N
    Ferrite.getcelltype(grid::SmallGrid) = eltype(grid.cells_test)
    Ferrite.getcelltype(grid::SmallGrid, i::Int) = typeof(grid.cells_test[i])
    Ferrite.get_node_coordinate(x::NTuple{dim,Float64}) where dim = Vec{dim,Float64}(x)

    Ferrite.getnodes(grid::SmallGrid) = grid.nodes_test
    Ferrite.getnodes(grid::SmallGrid, v::Union{Int, Vector{Int}}) = grid.nodes_test[v]
    Ferrite.getnnodes(grid::SmallGrid) = length(grid.nodes_test)
    Ferrite.get_coordinate_eltype(::SmallGrid) = Float64
    Ferrite.get_coordinate_type(::SmallGrid{dim}) where dim = Vec{dim,Float64}
    Ferrite.nnodes_per_cell(grid::SmallGrid, i::Int=1) = Ferrite.nnodes(grid.cells_test[i])
    Ferrite.n_faces_per_cell(grid::SmallGrid) = nfaces(eltype(grid.cells_test))

    nodes = [(-1.0,-1.0); (0.0,-1.0); (1.0,-1.0); (-1.0,0.0); (0.0,0.0); (1.0,0.0); (-1.0,1.0); (0.0,1.0); (1.0,1.0)]
    cells = (Quadrilateral((1,2,5,4)), Quadrilateral((2,3,6,5)), Quadrilateral((4,5,8,7)), Quadrilateral((5,6,9,8)))
    subtype_grid = SmallGrid(nodes,cells)
    reference_grid = generate_grid(Quadrilateral, (2,2))

    ip = Lagrange{RefQuadrilateral, 1}()
    qr = QuadratureRule{RefQuadrilateral}(2)
    cellvalues = CellValues(qr, ip);
    
    dhs = [DofHandler(grid) for grid in (subtype_grid, reference_grid)]
    u1 = Vector{Float64}(undef, 9)
    u2 = Vector{Float64}(undef, 9)
    ∂Ω = union(getfaceset.((reference_grid, ), ["left", "right", "top", "bottom"])...)
    dbc = Dirichlet(:u, ∂Ω, (x, t) -> 0)

    function doassemble!(cellvalues::CellValues, K::SparseMatrixCSC, dh::DofHandler)
        n_basefuncs = getnbasefunctions(cellvalues)
        Ke = zeros(n_basefuncs, n_basefuncs)
        fe = zeros(n_basefuncs)
        f = zeros(ndofs(dh))
        assembler = start_assemble(K, f)
        for cellid in 1:getncells(dh.grid)
            fill!(Ke, 0)
            fill!(fe, 0)
            coords = getcoordinates(dh.grid, cellid)
            reinit!(cellvalues, coords)
            for q_point in 1:getnquadpoints(cellvalues)
                dΩ = getdetJdV(cellvalues, q_point)
                for i in 1:n_basefuncs
                    v  = shape_value(cellvalues, q_point, i)
                    ∇v = shape_gradient(cellvalues, q_point, i)
                    fe[i] += v * dΩ
                    for j in 1:n_basefuncs
                        ∇u = shape_gradient(cellvalues, q_point, j)
                        Ke[i, j] += (∇v ⋅ ∇u) * dΩ
                    end
                end
            end
            assemble!(assembler, celldofs(dh,cellid), fe, Ke)
        end
        return K, f
    end

    for (dh,u) in zip(dhs,(u1,u2))
        add!(dh, :u, ip)
        close!(dh)
        ch = ConstraintHandler(dh)
        add!(ch, dbc)
        close!(ch)
        update!(ch, 0.0)
        K = create_matrix(dh);
        K, f = doassemble!(cellvalues, K, dh);
        apply!(K, f, ch)
        sol = K \ f
        u .= sol
    end

    @test Ferrite.ndofs_per_cell(dhs[1]) == Ferrite.ndofs_per_cell(dhs[2])
    @test Ferrite.celldofs(dhs[1],3) == Ferrite.celldofs(dhs[2],3)
    @test Ferrite.ndofs(dhs[1]) == Ferrite.ndofs(dhs[2])
    @test isapprox(u1,u2,atol=1e-8)

    minv, maxv = Ferrite.bounding_box(subtype_grid)
    @test minv ≈ Vec((-1.0,-1.0))
    @test maxv ≈ Vec((+1.0,+1.0))

    colors1 = Ferrite.create_coloring(subtype_grid, alg = ColoringAlgorithm.WorkStream)
    colors2 = Ferrite.create_coloring(reference_grid, alg = ColoringAlgorithm.WorkStream)
    @test all(colors1 .== colors2)

    colors1 = Ferrite.create_coloring(subtype_grid, alg = ColoringAlgorithm.Greedy)
    colors2 = Ferrite.create_coloring(reference_grid, alg = ColoringAlgorithm.Greedy)
    @test all(colors1 .== colors2)
end
