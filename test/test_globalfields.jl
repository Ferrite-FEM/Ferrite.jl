using Ferrite
using Test
using LinearAlgebra
using Random: MersenneTwister, shuffle
using SparseArrays
using OrderedCollections: OrderedSet

# Reference sparsity pattern computed with plain nested loops over the cell dofs, with
# optional field-level coupling and constraint elimination. Used to verify the (buffered)
# global dof row insertion in `_add_cell_entries!`.
function reference_pattern(dh; coupling = nothing, ch = nothing, keep_constrained = true)
    entries = Set{Tuple{Int, Int}}()
    for d in 1:ndofs(dh)
        push!(entries, (d, d))
    end
    prescribed = ch === nothing ? Set{Int}() : Set(ch.prescribed_dofs)
    for sdh in dh.subdofhandlers
        franges = [dof_range(sdh, name) for name in sdh.field_names]
        gidxs = [findfirst(==(name), dh.field_names)::Int for name in sdh.field_names]
        for cellnr in sdh.cellset
            cdofs = celldofs(dh, cellnr)
            for (fi, ri) in pairs(franges), (fj, rj) in pairs(franges)
                coupling === nothing || coupling[gidxs[fi], gidxs[fj]] || continue
                for I in ri, J in rj
                    row, col = cdofs[I], cdofs[J]
                    !keep_constrained && (row in prescribed || col in prescribed) && continue
                    push!(entries, (row, col))
                end
            end
        end
    end
    return entries
end

function pattern_entries(sp)
    entries = Set{Tuple{Int, Int}}()
    for (row, colidxs) in enumerate(Ferrite.eachrow(sp))
        for col in colidxs
            push!(entries, (row, col))
        end
    end
    return entries
end

# Grid with 2 quadrilaterals (bottom) and 4 triangles (top)
function mixed_grid()
    nodes = Node.(
        [
            Vec((0.0, 0.0)), Vec((1.0, 0.0)), Vec((2.0, 0.0)),
            Vec((0.0, 1.0)), Vec((1.0, 1.0)), Vec((2.0, 1.0)),
            Vec((0.0, 2.0)), Vec((1.0, 2.0)), Vec((2.0, 2.0)),
        ]
    )
    cells = [
        Quadrilateral((1, 2, 5, 4)),
        Quadrilateral((2, 3, 6, 5)),
        Triangle((4, 5, 8)),
        Triangle((4, 8, 7)),
        Triangle((5, 6, 9)),
        Triangle((5, 9, 8)),
    ]
    return Grid(cells, nodes)
end

# Wrapper interpolation in the style of FerriteViz.jl's MatrixizedInterpolation: it
# forwards InterpolationInfo (and getnbasefunctions) but NOT the individual dof-index
# accessor functions (vertexdof_indices etc.). The classification check in close! must
# accept it (regression test for the downstream FerriteViz failure).
struct MatrixizedTestInterpolation{vdim1, vdim2, refshape, order, SI <: ScalarInterpolation{refshape, order}} <: Ferrite.Interpolation{refshape, order}
    ip::SI
    function MatrixizedTestInterpolation{vdim1, vdim2}(ip::SI) where {vdim1, vdim2, refshape, order, SI <: Ferrite.ScalarInterpolation{refshape, order}}
        return new{vdim1, vdim2, refshape, order, SI}(ip)
    end
end
Ferrite.n_components(::MatrixizedTestInterpolation{vdim1, vdim2}) where {vdim1, vdim2} = vdim1 * vdim2
Ferrite.adjust_dofs_during_distribution(ip::MatrixizedTestInterpolation) = Ferrite.adjust_dofs_during_distribution(ip.ip)
Ferrite.InterpolationInfo(ip::MatrixizedTestInterpolation{vdim1, vdim2}) where {vdim1, vdim2} = Ferrite.InterpolationInfo(ip.ip, vdim1 * vdim2)
Ferrite.getnbasefunctions(ip::MatrixizedTestInterpolation{vdim1, vdim2}) where {vdim1, vdim2} = vdim1 * vdim2 * getnbasefunctions(ip.ip)

# Scalar interpolation whose dof-index accessors do not classify all base functions;
# close! must reject it.
struct HalfClassifiedTestInterpolation <: ScalarInterpolation{RefTriangle, 1} end
Ferrite.getnbasefunctions(::HalfClassifiedTestInterpolation) = 4
Ferrite.vertexdof_indices(::HalfClassifiedTestInterpolation) = ((1,), (2,), (3,))
Ferrite.adjust_dofs_during_distribution(::HalfClassifiedTestInterpolation) = false

@testset "Global fields (GlobalConstant)" begin

    @testset "dof distribution" begin
        # Single SubDofHandler, scalar global field
        grid = generate_grid(Triangle, (3, 3))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefTriangle, 1}())
        add!(dh, :λ, GlobalConstant{RefTriangle}())
        close!(dh)
        @test ndofs(dh) == getnnodes(grid) + 1
        @test ndofs_per_cell(dh) == 4
        gdofs = global_field_dofs(dh, :λ)
        @test length(gdofs) == 1
        for cell in CellIterator(dh)
            # Global field dofs are part of the celldofs of every cell, at dof_range
            @test celldofs(cell)[dof_range(dh, :λ)] == gdofs
        end
        # Not a global field / not a field at all
        @test_throws ErrorException global_field_dofs(dh, :u)
        @test_throws ErrorException global_field_dofs(dh, :nofield)

        # Global field ordered *between* regular fields
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefTriangle, 1}())
        add!(dh, :λ, GlobalConstant{RefTriangle}())
        add!(dh, :p, Lagrange{RefTriangle, 1}())
        close!(dh)
        @test ndofs(dh) == 2 * getnnodes(grid) + 1
        @test dof_range(dh, :u) == 1:3
        @test dof_range(dh, :λ) == 4:4
        @test dof_range(dh, :p) == 5:7
        for cell in CellIterator(dh)
            @test celldofs(cell)[dof_range(dh, :λ)] == global_field_dofs(dh, :λ)
        end

        # Vectorized global field: dofs are consecutive and in component order
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefTriangle, 1}()^2)
        add!(dh, :εv, GlobalConstant{RefTriangle}()^3)
        close!(dh)
        @test ndofs(dh) == 2 * getnnodes(grid) + 3
        gdofs = global_field_dofs(dh, :εv)
        @test length(gdofs) == 3
        @test gdofs == gdofs[1]:gdofs[3] # consecutive after close!
        for cell in CellIterator(dh)
            @test celldofs(cell)[dof_range(dh, :εv)] == gdofs
        end

        # Multiple independent global fields with overlapping/differing supports, and
        # different local field orderings in different SubDofHandlers
        grid = generate_grid(Quadrilateral, (4, 4))
        set_a = OrderedSet(1:8)
        set_b = OrderedSet(9:16)
        dh = DofHandler(grid)
        sdh_a = SubDofHandler(dh, set_a)
        add!(sdh_a, :u, Lagrange{RefQuadrilateral, 1}())
        add!(sdh_a, :λ, GlobalConstant{RefQuadrilateral}())
        add!(sdh_a, :μ, GlobalConstant{RefQuadrilateral}()^2)
        sdh_b = SubDofHandler(dh, set_b)
        add!(sdh_b, :λ, GlobalConstant{RefQuadrilateral}()) # different order than sdh_a
        add!(sdh_b, :u, Lagrange{RefQuadrilateral, 1}())
        close!(dh)
        @test ndofs(dh) == getnnodes(grid) + 1 + 2
        λdofs = global_field_dofs(dh, :λ)
        μdofs = global_field_dofs(dh, :μ)
        @test isempty(intersect(λdofs, μdofs))
        # :λ shares dofs between both SubDofHandlers, :μ only lives on sdh_a
        for cell in CellIterator(sdh_a)
            @test celldofs(cell)[dof_range(sdh_a, :λ)] == λdofs
            @test celldofs(cell)[dof_range(sdh_a, :μ)] == μdofs
        end
        for cell in CellIterator(sdh_b)
            @test celldofs(cell)[dof_range(sdh_b, :λ)] == λdofs
        end
        # :u is continuous across the sdh_a/sdh_b partition (entity dofs shared by name):
        # each node has exactly one dof
        node_dofs = Dict{Int, Int}()
        udofs_total = Set{Int}()
        for sdh in dh.subdofhandlers, cell in CellIterator(sdh)
            for (node, dof) in zip(getnodes(cell), celldofs(cell)[dof_range(sdh, :u)])
                @test get!(node_dofs, node, dof) == dof
                push!(udofs_total, dof)
            end
        end
        @test length(udofs_total) == getnnodes(grid)

        # Same global field on different reference shapes (mixed grid)
        grid = mixed_grid()
        dh = DofHandler(grid)
        sdh_quad = SubDofHandler(dh, OrderedSet(1:2))
        add!(sdh_quad, :u, Lagrange{RefQuadrilateral, 1}())
        add!(sdh_quad, :λ, GlobalConstant{RefQuadrilateral}())
        sdh_tri = SubDofHandler(dh, OrderedSet(3:6))
        add!(sdh_tri, :u, Lagrange{RefTriangle, 1}())
        add!(sdh_tri, :λ, GlobalConstant{RefTriangle}())
        close!(dh)
        @test ndofs(dh) == getnnodes(grid) + 1
        λdofs = global_field_dofs(dh, :λ)
        for sdh in dh.subdofhandlers, cell in CellIterator(sdh)
            @test celldofs(cell)[dof_range(sdh, :λ)] == λdofs
        end

        # Refshape compatibility is checked like for any interpolation
        dh = DofHandler(generate_grid(Triangle, (2, 2)))
        @test_throws ErrorException add!(dh, :λ, GlobalConstant{RefQuadrilateral}())

        # A field cannot be global in one SubDofHandler and regular in another
        grid = generate_grid(Quadrilateral, (2, 2))
        dh = DofHandler(grid)
        sdh1 = SubDofHandler(dh, OrderedSet(1:2))
        add!(sdh1, :p, GlobalConstant{RefQuadrilateral}())
        sdh2 = SubDofHandler(dh, OrderedSet(3:4))
        @test_throws ErrorException add!(sdh2, :p, Lagrange{RefQuadrilateral, 1}())
        # ... also through vectorization
        dh = DofHandler(grid)
        sdh1 = SubDofHandler(dh, OrderedSet(1:2))
        add!(sdh1, :p, GlobalConstant{RefQuadrilateral}()^2)
        sdh2 = SubDofHandler(dh, OrderedSet(3:4))
        @test_throws ErrorException add!(sdh2, :p, Lagrange{RefQuadrilateral, 1}()^2)
    end

    @testset "interpolation interface" begin
        ip = GlobalConstant{RefTriangle}()
        @test getnbasefunctions(ip) == 1
        @test Ferrite.getorder(ip) == 0
        @test Ferrite.n_components(ip) == 1
        @test Ferrite.n_components(ip^3) == 3
        @test Ferrite.is_global_field_interpolation(ip)
        @test Ferrite.is_global_field_interpolation(ip^3)
        @test !Ferrite.is_global_field_interpolation(Lagrange{RefTriangle, 1}())
        @test !Ferrite.is_global_field_interpolation(Lagrange{RefTriangle, 1}()^2)
        @test Ferrite.conformity(ip) == Ferrite.H1Conformity()
        @test Ferrite.reference_shape_value(ip, Vec((0.1, 0.2)), 1) === 1.0
        @test_throws ArgumentError Ferrite.reference_shape_value(ip, Vec((0.1, 0.2)), 2)
        # CellValues works (shape functions constant 1, zero gradient)
        qr = QuadratureRule{RefTriangle}(2)
        cv = CellValues(qr, ip, Lagrange{RefTriangle, 1}())
        grid = generate_grid(Triangle, (1, 1))
        reinit!(cv, getcoordinates(grid, 1))
        @test shape_value(cv, 1, 1) == 1.0
        @test norm(shape_gradient(cv, 1, 1)) == 0.0
        cvv = CellValues(qr, ip^2, Lagrange{RefTriangle, 1}())
        reinit!(cvv, getcoordinates(grid, 1))
        @test shape_value(cvv, 1, 1) == Vec((1.0, 0.0))
        @test shape_value(cvv, 1, 2) == Vec((0.0, 1.0))
    end

    @testset "close! base function classification" begin
        grid = generate_grid(Triangle, (3, 3))
        # FerriteViz-style wrapper interpolation (only InterpolationInfo forwarded):
        # 3 basefns × 4 copies, all volume dofs (discontinuous)
        dh = DofHandler(grid)
        add!(dh, :gradient, MatrixizedTestInterpolation{2, 2}(DiscontinuousLagrange{RefTriangle, 1}()))
        close!(dh)
        @test ndofs(dh) == getncells(grid) * 3 * 4
        @test ndofs_per_cell(dh) == 12
        # An interpolation that does not classify all base functions is rejected
        dh = DofHandler(grid)
        add!(dh, :u, HalfClassifiedTestInterpolation())
        @test_throws AssertionError close!(dh)
    end

    @testset "sparsity pattern" begin
        grid = generate_grid(Quadrilateral, (5, 5))
        ipu = Lagrange{RefQuadrilateral, 1}()^2
        ipλ = GlobalConstant{RefQuadrilateral}()^2
        dh = DofHandler(grid)
        add!(dh, :u, ipu)
        add!(dh, :λ, ipλ)
        close!(dh)

        # Full coupling equals the brute force reference
        sp = Ferrite.init_sparsity_pattern(dh)
        Ferrite.add_sparsity_entries!(sp, dh)
        @test pattern_entries(sp) == reference_pattern(dh)

        # FastSparsityPattern (allocate_matrix(dh) fast path) gives the same matrix
        K = allocate_matrix(dh)
        Kref = Ferrite.allocate_matrix(sp)
        @test K.colptr == Kref.colptr && K.rowval == Kref.rowval

        # Asymmetric field coupling: λ-rows couple to u, but u-rows not to λ
        coupling = [true true; false true] # [uu uλ; λu λλ] with fields (u, λ): row=test fn
        # Note: coupling[i, j] refers to (field i, field j) block, i.e. rows of field i
        sp = Ferrite.init_sparsity_pattern(dh)
        Ferrite.add_sparsity_entries!(sp, dh; coupling = [true false; true true])
        @test pattern_entries(sp) == reference_pattern(dh; coupling = [true false; true true])
        sp = Ferrite.init_sparsity_pattern(dh)
        Ferrite.add_sparsity_entries!(sp, dh; coupling = coupling)
        @test pattern_entries(sp) == reference_pattern(dh; coupling = coupling)

        # Pre-populated pattern: entries survive and dedup works
        sp = Ferrite.init_sparsity_pattern(dh)
        Ferrite.add_entry!(sp, 1, ndofs(dh))
        Ferrite.add_entry!(sp, ndofs(dh), 1)
        gdof = first(global_field_dofs(dh, :λ))
        Ferrite.add_entry!(sp, gdof, 2) # entry that the cell loop adds again
        Ferrite.add_sparsity_entries!(sp, dh)
        @test pattern_entries(sp) == union(reference_pattern(dh), Set([(1, ndofs(dh)), (ndofs(dh), 1)]))

        # keep_constrained = false with constrained regular and global dofs
        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:u, getfacetset(grid, "left"), x -> zero(Vec{2})))
        add!(ch, AffineConstraint(gdof, Pair{Int, Float64}[], 0.0)) # ground one global dof
        close!(ch)
        sp = Ferrite.init_sparsity_pattern(dh)
        Ferrite.add_sparsity_entries!(sp, dh, ch; keep_constrained = false)
        entries = pattern_entries(sp)
        @test entries == reference_pattern(dh; ch = ch, keep_constrained = false)
        # the grounded global dof only has its diagonal entry
        @test all(e -> !(gdof in e) || e == (gdof, gdof), entries)

        # BlockSparsityPattern goes through the same generic code
        renumber!(dh, DofOrder.FieldWise()) # block pattern requires blocked dofs
        gdofs = global_field_dofs(dh, :λ)
        @test gdofs == collect((ndofs(dh) - 1):ndofs(dh)) # λ block last after renumbering
        nudofs = ndofs(dh) - 2
        bsp = BlockSparsityPattern([nudofs, 2])
        Ferrite.add_sparsity_entries!(bsp, dh)
        @test pattern_entries(bsp) == reference_pattern(dh)

        # AffineConstraint with a global dof as master: condensation adds entries between
        # the master (global) dof and the neighbors of the dependent dof
        dh = DofHandler(grid)
        add!(dh, :u, ipu)
        add!(dh, :λ, ipλ)
        close!(dh)
        gdofs = global_field_dofs(dh, :λ)
        # constrain some u dof to the first global dof
        udof = first(celldofs(dh, 13)[dof_range(dh, :u)])
        ch = ConstraintHandler(dh)
        add!(ch, AffineConstraint(udof, [gdofs[1] => 1.0], 0.0))
        close!(ch)
        sp = Ferrite.init_sparsity_pattern(dh)
        Ferrite.add_sparsity_entries!(sp, dh, ch)
        entries = pattern_entries(sp)
        for (row, col) in reference_pattern(dh)
            if col == udof
                @test (row, gdofs[1]) in entries
            end
        end
        K = Ferrite.allocate_matrix(sp)
        f = zeros(ndofs(dh))
        apply!(K, f, ch) # smoke: condensation works in-place

        # Interface entries: global dofs are identical on both sides of an interface and
        # are correctly skipped (they couple through the cell entries). Documented
        # invariant: interface_coupling for a global field requires the corresponding
        # cell coupling (which is where global dofs actually couple).
        dgrid = generate_grid(Quadrilateral, (3, 3))
        dh_dg = DofHandler(dgrid)
        add!(dh_dg, :u, DiscontinuousLagrange{RefQuadrilateral, 1}())
        add!(dh_dg, :λ, GlobalConstant{RefQuadrilateral}())
        close!(dh_dg)
        topology = ExclusiveTopology(dgrid)
        full = trues(2, 2)
        sp = Ferrite.init_sparsity_pattern(dh_dg)
        Ferrite.add_sparsity_entries!(sp, dh_dg; topology, interface_coupling = full)
        entries = pattern_entries(sp)
        # cell entries cover all (λ, u) couplings already
        @test issubset(reference_pattern(dh_dg), entries)
        # the asymmetric corner: interface coupling enabled for λ but cell coupling
        # disabled -> no λ entries (except diagonal) since interface skips shared dofs
        no_λ = [true false; false false]
        sp = Ferrite.init_sparsity_pattern(dh_dg)
        Ferrite.add_sparsity_entries!(sp, dh_dg; coupling = no_λ, interface_coupling = full)
        gdof_dg = only(global_field_dofs(dh_dg, :λ))
        for (row, col) in pattern_entries(sp)
            if row == gdof_dg || col == gdof_dg
                @test row == col == gdof_dg
            end
        end
    end

    @testset "oversized global row (integrated)" begin
        # A global dof row with more columns than fit in a single regular PoolAllocator
        # page (4 MiB = 524288 Int columns) exercises the oversized page support through
        # the full sparsity pattern code path.
        grid = generate_grid(Quadrilateral, (520, 520))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefQuadrilateral, 1}()^2)
        add!(dh, :λ, GlobalConstant{RefQuadrilateral}())
        close!(dh)
        @test ndofs(dh) > Ferrite.PoolAllocator.PAGE_SIZE ÷ sizeof(Int)
        sp = Ferrite.init_sparsity_pattern(dh)
        Ferrite.add_sparsity_entries!(sp, dh)
        gdof = only(global_field_dofs(dh, :λ))
        # the global dof couples with everything (row and column)
        @test length(Ferrite.eachrow(sp, gdof)) == ndofs(dh)
        K = Ferrite.allocate_matrix(sp)
        @test length(nzrange(K, gdof)) == ndofs(dh)
    end

    @testset "export on subdomain support" begin
        # Global field only on the left half: constant on the support, NaN outside
        grid = generate_grid(Quadrilateral, (4, 4))
        left = OrderedSet([i + 4 * j for j in 0:3 for i in 1:2])
        right = OrderedSet(setdiff(1:16, left))
        dh = DofHandler(grid)
        sdh_l = SubDofHandler(dh, left)
        add!(sdh_l, :u, Lagrange{RefQuadrilateral, 1}())
        add!(sdh_l, :λ, GlobalConstant{RefQuadrilateral}())
        sdh_r = SubDofHandler(dh, right)
        add!(sdh_r, :u, Lagrange{RefQuadrilateral, 1}())
        close!(dh)
        u = rand(ndofs(dh))
        gdof = only(global_field_dofs(dh, :λ))
        vals = evaluate_at_grid_nodes(dh, u, :λ)
        support_nodes = Set(n for cellnr in left for n in getcells(grid, cellnr).nodes)
        for (node, val) in pairs(vals)
            if node in support_nodes
                @test val ≈ u[gdof]
            else
                @test isnan(val)
            end
        end
        mktempdir() do dir
            VTKGridFile(joinpath(dir, "subdomain"), dh) do vtk
                write_solution(vtk, dh, u)
            end
            @test isfile(joinpath(dir, "subdomain.vtu"))
        end
    end

    @testset "end-to-end: pure Neumann Poisson with mean value constraint" begin
        # -Δu = f in Ω, ∂u/∂n = 0 on ∂Ω, ∫u dΩ = 0 enforced with a global multiplier:
        #   (∇δu, ∇u) + (δu, λ) = (δu, f)
        #   (μ, u)               = 0
        # and compared against the AffineConstraint (dense row elimination) approach.
        grid = generate_grid(Quadrilateral, (10, 10))
        ip = Lagrange{RefQuadrilateral, 1}()
        qr = QuadratureRule{RefQuadrilateral}(2)
        f_fun(x) = sin(π * x[1]) * x[2]

        function assemble_local!(Ke, fe, cvu, cvλ, coords, ru, rλ)
            fill!(Ke, 0); fill!(fe, 0)
            for qp in 1:getnquadpoints(cvu)
                dΩ = getdetJdV(cvu, qp)
                x = spatial_coordinate(cvu, qp, coords)
                for (iu, I) in pairs(ru)
                    δu = shape_value(cvu, qp, iu)
                    fe[I] += f_fun(x) * δu * dΩ
                    for (ju, J) in pairs(ru)
                        Ke[I, J] += (shape_gradient(cvu, qp, iu) ⋅ shape_gradient(cvu, qp, ju)) * dΩ
                    end
                    for (jλ, J) in pairs(rλ)
                        μ = shape_value(cvλ, qp, jλ)
                        Ke[I, J] += δu * μ * dΩ
                        Ke[J, I] += δu * μ * dΩ
                    end
                end
            end
            return
        end

        # Global field version
        dh = DofHandler(grid)
        add!(dh, :u, ip)
        add!(dh, :λ, GlobalConstant{RefQuadrilateral}())
        close!(dh)
        cvu = CellValues(qr, ip)
        cvλ = CellValues(qr, GlobalConstant{RefQuadrilateral}(), ip)
        K = allocate_matrix(dh)
        f = zeros(ndofs(dh))
        assembler = start_assemble(K, f)
        n = ndofs_per_cell(dh)
        Ke = zeros(n, n); fe = zeros(n)
        ru, rλ = dof_range(dh, :u), dof_range(dh, :λ)
        for cell in CellIterator(dh)
            reinit!(cvu, cell); reinit!(cvλ, cell)
            assemble_local!(Ke, fe, cvu, cvλ, getcoordinates(cell), ru, rλ)
            assemble!(assembler, celldofs(cell), Ke, fe)
        end
        @test norm(K - K') == 0 # saddle point system is symmetric
        u = K \ f
        gdof = only(global_field_dofs(dh, :λ))

        # Reference: same problem with the mean value constraint as AffineConstraint
        # u_1 = -(∑_i w_i u_i)/w_1 where w = ∫ N dΩ
        dh_ref = DofHandler(grid)
        add!(dh_ref, :u, ip)
        close!(dh_ref)
        w = zeros(ndofs(dh_ref))
        for cell in CellIterator(dh_ref)
            reinit!(cvu, cell)
            for qp in 1:getnquadpoints(cvu), i in 1:getnbasefunctions(cvu)
                w[celldofs(cell)[i]] += shape_value(cvu, qp, i) * getdetJdV(cvu, qp)
            end
        end
        ch_ref = ConstraintHandler(dh_ref)
        add!(ch_ref, AffineConstraint(1, [i => -w[i] / w[1] for i in 2:ndofs(dh_ref)], 0.0))
        close!(ch_ref)
        Kr = allocate_matrix(dh_ref, ch_ref)
        fr = zeros(ndofs(dh_ref))
        assembler = start_assemble(Kr, fr)
        nu = getnbasefunctions(cvu)
        Ke_ref = zeros(nu, nu); fe_ref = zeros(nu)
        for cell in CellIterator(dh_ref)
            reinit!(cvu, cell)
            fill!(Ke_ref, 0); fill!(fe_ref, 0)
            for qp in 1:getnquadpoints(cvu)
                dΩ = getdetJdV(cvu, qp)
                x = spatial_coordinate(cvu, qp, getcoordinates(cell))
                for i in 1:nu
                    fe_ref[i] += f_fun(x) * shape_value(cvu, qp, i) * dΩ
                    for j in 1:nu
                        Ke_ref[i, j] += (shape_gradient(cvu, qp, i) ⋅ shape_gradient(cvu, qp, j)) * dΩ
                    end
                end
            end
            assemble!(assembler, celldofs(cell), Ke_ref, fe_ref)
        end
        apply!(Kr, fr, ch_ref)
        ur = Kr \ fr
        apply!(ur, ch_ref)

        # Same u solution (u dofs of dh and dh_ref are numbered identically since :u was
        # added first and λ dofs are appended per cell after the u dofs)
        udofs = setdiff(1:ndofs(dh), gdof)
        @test u[udofs] ≈ ur atol = 1.0e-10
        # The multiplier vanishes when ∫f dΩ = 0... it doesn't here, so just sanity check
        # that λ equals the mean of the source: λ = ∫f dΩ / |Ω|
        intf, vol = 0.0, 0.0
        for cell in CellIterator(dh)
            reinit!(cvu, cell)
            for qp in 1:getnquadpoints(cvu)
                intf += f_fun(spatial_coordinate(cvu, qp, getcoordinates(cell))) * getdetJdV(cvu, qp)
                vol += getdetJdV(cvu, qp)
            end
        end
        @test u[gdof] ≈ intf / vol atol = 1.0e-10

        # export: evaluate_at_grid_nodes gives the constant, write_solution works for
        # both the continuous and discontinuous vtk-grid paths
        λvals = evaluate_at_grid_nodes(dh, u, :λ)
        @test all(v -> v ≈ u[gdof], λvals)
        mktempdir() do dir
            VTKGridFile(joinpath(dir, "globalfield"), dh) do vtk
                write_solution(vtk, dh, u)
            end
            @test isfile(joinpath(dir, "globalfield.vtu"))
            VTKGridFile(joinpath(dir, "globalfield_dg"), dh; write_discontinuous = true) do vtk
                write_solution(vtk, dh, u)
            end
            @test isfile(joinpath(dir, "globalfield_dg.vtu"))
        end

        # PointEvalHandler returns the constant
        ph = PointEvalHandler(grid, [Vec(0.13, -0.87)])
        @test only(evaluate_at_points(ph, dh, u, :λ)) ≈ u[gdof]
    end

    @testset "constraints" begin
        grid = generate_grid(Triangle, (3, 3))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefTriangle, 1}()^2)
        add!(dh, :λ, GlobalConstant{RefTriangle}()^2)
        close!(dh)
        gdofs = global_field_dofs(dh, :λ)

        # Dirichlet-type conditions on global fields error in all three entry points
        ch = ConstraintHandler(dh)
        @test_throws ErrorException add!(ch, Dirichlet(:λ, getfacetset(grid, "left"), x -> 0.0))
        @test_throws ErrorException add!(ch, PeriodicDirichlet(:λ, ["left" => "right"]))
        @test_throws ErrorException add!(ch, ProjectedDirichlet(:λ, getfacetset(grid, "left"), (x, t, n) -> 0.0))

        # Dirichlet on the companion field in the same (sub)dofhandler works
        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:u, getfacetset(grid, "left"), x -> zero(Vec{2})))
        # AffineConstraint: global dof as dependent (grounding) and as master
        add!(ch, AffineConstraint(gdofs[2], Pair{Int, Float64}[], 0.0))
        udof = celldofs(dh, 5)[first(dof_range(dh, :u))]
        add!(ch, AffineConstraint(udof, [gdofs[1] => 2.0], 0.0))
        close!(ch)
        K = allocate_matrix(dh, ch)
        f = zeros(ndofs(dh))
        for cell in CellIterator(dh)
            n = length(celldofs(cell))
            assemble!(start_assemble(K, f; fillzero = false), celldofs(cell), Matrix(1.0I, n, n), ones(n))
        end
        apply!(K, f, ch)
        u = K \ f
        apply!(u, ch)
        @test u[gdofs[2]] == 0.0
        @test u[udof] ≈ 2.0 * u[gdofs[1]]

        # apply_analytical! rejects global fields (scalar and vector)
        a = zeros(ndofs(dh))
        apply_analytical!(a, dh, :u, x -> zero(Vec{2})) # companion field works
        @test_throws ErrorException apply_analytical!(a, dh, :λ, x -> zero(Vec{2}))
        dhs = DofHandler(grid)
        add!(dhs, :λ, GlobalConstant{RefTriangle}())
        close!(dhs)
        @test_throws ErrorException apply_analytical!(zeros(ndofs(dhs)), dhs, :λ, x -> 0.0)
    end

    @testset "renumbering" begin
        grid = generate_grid(Quadrilateral, (3, 3))
        dh = DofHandler(grid)
        add!(dh, :λ, GlobalConstant{RefQuadrilateral}()^3)
        add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
        close!(dh)
        gdofs_orig = global_field_dofs(dh, :λ)
        @test gdofs_orig == [1, 2, 3] # distributed first for the first cell

        # FieldWise: λ block first (field order), u block second
        renumber!(dh, DofOrder.FieldWise())
        @test global_field_dofs(dh, :λ) == [1, 2, 3]
        # FieldWise with target blocks: move λ last
        renumber!(dh, DofOrder.FieldWise([2, 1]))
        @test global_field_dofs(dh, :λ) == collect((ndofs(dh) - 2):ndofs(dh))
        # ComponentWise blocks the three λ components separately, but relative component
        # order is maintained
        renumber!(dh, DofOrder.ComponentWise())
        @test global_field_dofs(dh, :λ) == [1, 2, 3]

        # Arbitrary permutation making the vector global field non-contiguous: the
        # accessor preserves component order
        perm = collect(ndofs(dh):-1:1)
        renumber!(dh, perm)
        gdofs = global_field_dofs(dh, :λ)
        @test gdofs == [ndofs(dh), ndofs(dh) - 1, ndofs(dh) - 2]
        for cell in CellIterator(dh)
            @test celldofs(cell)[dof_range(dh, :λ)] == gdofs
        end

        # Solution invariance: assembling and solving after renumbering gives the same
        # physical solution (nodal u values and λ components)
        function assemble_and_solve(dh)
            ip = Lagrange{RefQuadrilateral, 1}()
            ipλ = GlobalConstant{RefQuadrilateral}()^3
            qr = QuadratureRule{RefQuadrilateral}(2)
            cvu = CellValues(qr, ip)
            cvλ = CellValues(qr, ipλ, ip)
            K = allocate_matrix(dh)
            f = zeros(ndofs(dh))
            assembler = start_assemble(K, f)
            n = ndofs_per_cell(dh)
            Ke = zeros(n, n); fe = zeros(n)
            ru, rλ = dof_range(dh, :u), dof_range(dh, :λ)
            for cell in CellIterator(dh)
                reinit!(cvu, cell); reinit!(cvλ, cell)
                fill!(Ke, 0); fill!(fe, 0)
                for qp in 1:getnquadpoints(cvu)
                    dΩ = getdetJdV(cvu, qp)
                    x = spatial_coordinate(cvu, qp, getcoordinates(cell))
                    for (iu, I) in pairs(ru)
                        δu = shape_value(cvu, qp, iu)
                        fe[I] += (x[1] + x[2]^2) * δu * dΩ
                        for (ju, J) in pairs(ru)
                            Ke[I, J] += (shape_gradient(cvu, qp, iu) ⋅ shape_gradient(cvu, qp, ju) + δu * shape_value(cvu, qp, ju)) * dΩ
                        end
                        for (jλ, J) in pairs(rλ)
                            μ = shape_value(cvλ, qp, jλ) ⋅ Vec((1.0, 2.0, 3.0))
                            Ke[I, J] += δu * μ * dΩ
                            Ke[J, I] += δu * μ * dΩ
                        end
                    end
                    for (iλ, I) in pairs(rλ), (jλ, J) in pairs(rλ)
                        Ke[I, J] += (shape_value(cvλ, qp, iλ) ⋅ shape_value(cvλ, qp, jλ)) * dΩ
                    end
                end
                assemble!(assembler, celldofs(cell), Ke, fe)
            end
            u = K \ f
            return evaluate_at_grid_nodes(dh, u, :u), u[global_field_dofs(dh, :λ)]
        end
        dh = DofHandler(grid)
        add!(dh, :λ, GlobalConstant{RefQuadrilateral}()^3)
        add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
        close!(dh)
        u_nodes, λ = assemble_and_solve(dh)
        renumber!(dh, DofOrder.FieldWise([2, 1])) # λ block moved last
        u_nodes2, λ2 = assemble_and_solve(dh)
        @test u_nodes ≈ u_nodes2
        @test λ ≈ λ2
        renumber!(dh, shuffle(MersenneTwister(1234), collect(1:ndofs(dh))))
        u_nodes3, λ3 = assemble_and_solve(dh)
        @test u_nodes ≈ u_nodes3
        @test λ ≈ λ3
    end

    @testset "threaded assembly recipe" begin
        # Colored threaded assembly is UNSAFE with global fields: every cell of the
        # SubDofHandler shares the global dofs, so no valid coloring exists. This test
        # demonstrates (and checks) the safe recipe: within each color, writes to
        # K[i, j] for i, j both non-global are safe as usual, and entries where exactly
        # one index is global are written only by cells owning the other (non-shared)
        # index, which the coloring separates. The remaining conflicting contributions —
        # the global-global block and the global rows of the rhs — are accumulated in
        # task-local buffers and reduced serially after the loop.
        grid = generate_grid(Quadrilateral, (20, 20))
        ip = Lagrange{RefQuadrilateral, 1}()
        ipλ = GlobalConstant{RefQuadrilateral}()^2
        qr = QuadratureRule{RefQuadrilateral}(2)
        dh = DofHandler(grid)
        add!(dh, :u, ip)
        add!(dh, :λ, ipλ)
        close!(dh)
        ru, rλ = dof_range(dh, :u), dof_range(dh, :λ)
        gdofs = global_field_dofs(dh, :λ)
        nλ = length(gdofs)
        n = ndofs_per_cell(dh)

        function local_matrix!(Ke, fe, cvu, cvλ, ru, rλ)
            fill!(Ke, 0); fill!(fe, 0)
            for qp in 1:getnquadpoints(cvu)
                dΩ = getdetJdV(cvu, qp)
                for (iu, I) in pairs(ru)
                    δu = shape_value(cvu, qp, iu)
                    fe[I] += δu * dΩ
                    for (ju, J) in pairs(ru)
                        Ke[I, J] += (shape_gradient(cvu, qp, iu) ⋅ shape_gradient(cvu, qp, ju) + δu * shape_value(cvu, qp, ju)) * dΩ
                    end
                    for (jλ, J) in pairs(rλ)
                        μ = shape_value(cvλ, qp, jλ) ⋅ Vec((1.0, 2.0))
                        Ke[I, J] += δu * μ * dΩ
                        Ke[J, I] += δu * μ * dΩ
                    end
                end
                for (iλ, I) in pairs(rλ), (jλ, J) in pairs(rλ)
                    Ke[I, J] += (shape_value(cvλ, qp, iλ) ⋅ shape_value(cvλ, qp, jλ)) * dΩ
                    fe[I] += (shape_value(cvλ, qp, iλ) ⋅ Vec((1.0, 1.0))) * dΩ
                end
            end
            return
        end

        # Serial reference
        K_serial = allocate_matrix(dh)
        f_serial = zeros(ndofs(dh))
        let assembler = start_assemble(K_serial, f_serial),
                cvu = CellValues(qr, ip), cvλ = CellValues(qr, ipλ, ip),
                Ke = zeros(n, n), fe = zeros(n)
            for cell in CellIterator(dh)
                reinit!(cvu, cell); reinit!(cvλ, cell)
                local_matrix!(Ke, fe, cvu, cvλ, ru, rλ)
                assemble!(assembler, celldofs(cell), Ke, fe)
            end
        end

        # Threaded with per-task reduction of the conflicting contributions
        K = allocate_matrix(dh)
        f = zeros(ndofs(dh))
        colors = create_coloring(grid)
        assembler = start_assemble(K, f)
        nt = Threads.nthreads()
        # task local scratch + accumulation buffers for the global-global block and the
        # global rhs entries
        Kλλ_buffers = [zeros(nλ, nλ) for _ in 1:nt]
        fλ_buffers = [zeros(nλ) for _ in 1:nt]
        for color in colors
            chunks = Iterators.partition(color, max(1, cld(length(color), nt)))
            tasks = map(enumerate(chunks)) do (tid, chunk)
                Threads.@spawn begin
                    cvu = CellValues(qr, ip)
                    cvλ = CellValues(qr, ipλ, ip)
                    Ke = zeros(n, n)
                    fe = zeros(n)
                    cc = CellCache(dh)
                    for cellnr in chunk
                        reinit!(cc, cellnr)
                        reinit!(cvu, cc); reinit!(cvλ, cc)
                        local_matrix!(Ke, fe, cvu, cvλ, ru, rλ)
                        # Move the conflicting contributions to the task-local buffers
                        Kλλ_buffers[$tid] .+= @view Ke[rλ, rλ]
                        Ke[rλ, rλ] .= 0
                        fλ_buffers[$tid] .+= @view fe[rλ]
                        fe[rλ] .= 0
                        assemble!(assembler, celldofs(cc), Ke, fe)
                    end
                end
            end
            foreach(wait, tasks)
        end
        # Serial reduction of the buffered contributions
        for tid in 1:nt
            for (i, I) in pairs(gdofs), (j, J) in pairs(gdofs)
                K[I, J] += Kλλ_buffers[tid][i, j]
            end
            for (i, I) in pairs(gdofs)
                f[I] += fλ_buffers[tid][i]
            end
        end
        @test K ≈ K_serial
        @test f ≈ f_serial
    end
end
