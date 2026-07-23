# Imports for parallel (isolated) test execution:
using LinearAlgebra

@testset "apply_analytical!" begin

    # Convenience helper functions
    typebase(T::DataType) = T.name.wrapper
    typebase(v) = typebase(typeof(v))
    change_ip_order(ip::Interpolation, ::Nothing) = ip
    function change_ip_order(ip::Interpolation, order::Int)
        B = typebase(ip)
        RefShape = Ferrite.getrefshape(ip)
        return B{RefShape, order}()
    end
    getcellorder(CT) = Ferrite.getorder(Ferrite.geometric_interpolation(CT))
    getcelltypedim(::Type{<:Ferrite.AbstractCell{shape}}) where {dim, shape <: Ferrite.AbstractRefShape{dim}} = dim

    # Functions to create dof handlers for testing
    function testdh(CT, ip_order_u, ip_order_p)
        dim = getcelltypedim(CT)
        local grid
        try
            grid = generate_grid(CT, ntuple(_ -> 3, dim))
        catch e
            isa(e, MethodError) && e.f == generate_grid && return nothing
            rethrow(e)
        end

        dh = DofHandler(grid)
        default_ip = Ferrite.geometric_interpolation(CT)
        try
            add!(dh, :u, change_ip_order(default_ip, ip_order_u)^dim)
            add!(dh, :p, change_ip_order(default_ip, ip_order_p))
        catch e
            isa(e, MethodError) && e.f == Ferrite.reference_coordinates && return nothing
            rethrow(e)
        end
        close!(dh)
        return dh
    end

    function testdh_subdomains(dim, ip_order_u, ip_order_p)
        if dim == 1
            nodes = Node.([Vec{1}((x,)) for x in 0.0:1.0:3.0])
            cell1 = Line((1, 2))
            cell2 = QuadraticLine((2, 3, 4))
            grid = Grid([cell1, cell2], nodes; cellsets = Dict("A" => Set(1:1), "B" => Set(2:2)))
        elseif dim == 2
            nodes = Node.([Vec{2}((x, y)) for y in 0.0:2 for x in 0.0:1])
            cell1 = Triangle((1, 2, 3))
            cell2 = Triangle((2, 4, 3))
            cell3 = Quadrilateral((3, 4, 6, 5))
            grid = Grid([cell1, cell2, cell3], nodes; cellsets = Dict("A" => Set(1:2), "B" => Set(3:3)))
        else
            error("Only dim=1 & 2 supported")
        end
        default_ip_A = Ferrite.geometric_interpolation(getcelltype(grid, first(getcellset(grid, "A"))))
        default_ip_B = Ferrite.geometric_interpolation(getcelltype(grid, first(getcellset(grid, "B"))))
        dh = DofHandler(grid)
        sdh_A = SubDofHandler(dh, getcellset(grid, "A"))
        add!(sdh_A, :u, change_ip_order(default_ip_A, ip_order_u)^dim)
        add!(sdh_A, :p, change_ip_order(default_ip_A, ip_order_p))
        sdh_B = SubDofHandler(dh, getcellset(grid, "B"))
        add!(sdh_B, :u, change_ip_order(default_ip_B, ip_order_u)^dim)
        close!(dh)
        return dh
    end

    # The following can be removed after #457 is merged if that will include the MixedDofHandler
    function _global_dof_range(dh::DofHandler, field_name::Symbol)
        dofs = Set{Int}()
        for sdh in dh.subdofhandlers
            if !isnothing(Ferrite._find_field(sdh, field_name))
                _global_dof_range!(dofs, sdh, field_name, sdh.cellset)
            end
        end
        return sort!(collect(Int, dofs))
    end

    function _global_dof_range!(dofs, sdh, field_name, cellset)
        eldofs = celldofs(sdh.dh, first(cellset))
        field_range = dof_range(sdh, field_name)
        for i in cellset
            celldofs!(eldofs, sdh.dh, i)
            for j in field_range
                @inbounds d = eldofs[j]
                d in dofs || push!(dofs, d)
            end
        end
    end

    @testset "DofHandler" begin
        for CT in (
                Line, QuadraticLine,
                Triangle, QuadraticTriangle,
                Quadrilateral, QuadraticQuadrilateral,
                Tetrahedron, QuadraticTetrahedron,
                Hexahedron, # SerendipityQuadraticHexahedron,
                Wedge, Pyramid,
            )
            for ip_order_u in 1:2
                for ip_order_p in 1:2
                    dh = testdh(CT, ip_order_u, ip_order_p)
                    isnothing(dh) && continue # generate_grid not supported for this CT, or reference_coordinates not defined
                    num_udofs = length(_global_dof_range(dh, :u))
                    num_pdofs = length(_global_dof_range(dh, :p))

                    # Test average value
                    a = zeros(ndofs(dh))
                    f(x) = ones(Ferrite.get_coordinate_type(dh.grid))
                    apply_analytical!(a, dh, :u, f)
                    @test sum(a) / length(a) ≈ num_udofs / (num_udofs + num_pdofs)

                    # If not super/subparametric, compare with ConstraintHandler and node set
                    if ip_order_u == ip_order_p == getcellorder(CT)
                        fill!(a, 0)
                        a_ch = copy(a)
                        fp(x) = norm(x)^2
                        fu(x::Vec{1}) = Vec{1}((sin(x[1]),))
                        fu(x::Vec{2}) = Vec{2}((sin(x[1]), cos(x[2])))
                        fu(x::Vec{3}) = Vec{3}((sin(x[1]), cos(x[2]), x[3] * (x[1] + x[2])))
                        ch = ConstraintHandler(dh)
                        add!(ch, Dirichlet(:p, Set(1:getnnodes(dh.grid)), (x, t) -> fp(x)))
                        add!(ch, Dirichlet(:u, Set(1:getnnodes(dh.grid)), (x, t) -> fu(x)))
                        close!(ch); update!(ch, 0.0)
                        apply!(a_ch, ch)

                        apply_analytical!(a, dh, :u, fu)
                        apply_analytical!(a, dh, :p, fp)

                        @test a ≈ a_ch
                    end
                end
            end
        end
    end

    @testset "DofHandler with subdomains" begin
        for dim in 1:2
            for ip_order_u in 1:2
                for ip_order_p in 1:2
                    dh = testdh_subdomains(dim, ip_order_u, ip_order_p)
                    num_udofs = length(_global_dof_range(dh, :u))
                    num_pdofs = length(_global_dof_range(dh, :p))

                    # Test average value
                    a = zeros(ndofs(dh))
                    f(x) = ones(Vec{dim})
                    apply_analytical!(a, dh, :u, f)
                    @test sum(a) / length(a) ≈ num_udofs / (num_udofs + num_pdofs)

                    # Repeat test with calls for both subdomains separately
                    a = zeros(ndofs(dh))
                    apply_analytical!(a, dh, :u, f, getcellset(dh.grid, "A"))
                    apply_analytical!(a, dh, :u, f, getcellset(dh.grid, "B"))
                    @test sum(a) / length(a) ≈ num_udofs / (num_udofs + num_pdofs)
                end
            end
        end
    end

    @testset "Exceptions" begin
        dh = testdh(Quadrilateral, 1, 1)
        @test_throws ErrorException apply_analytical!(zeros(ndofs(dh)), dh, :v, x -> 0.0)    # Missing field
        @test_throws ErrorException apply_analytical!(zeros(ndofs(dh)), dh, :u, x -> 0.0)    # Should be f(x)::Vec{2}

        mdh = testdh_subdomains(2, 1, 1)
        @test_throws ErrorException apply_analytical!(zeros(ndofs(mdh)), mdh, :v, x -> 0.0)  # Missing field
        @test_throws ErrorException apply_analytical!(zeros(ndofs(mdh)), mdh, :u, x -> 0.0)  # Should be f(x)::Vec{2}
    end

    @testset "Non-nodal interpolations (H(div)/H(curl))" begin
        # Affine (but non-trivial) node transformation: shearing keeps quads/hexes
        # as parallelograms/parallelepipeds so that the cell mappings stay affine.
        function affine_grid(CT, nel)
            dim = Ferrite.getrefdim(CT)
            grid = generate_grid(CT, ntuple(Returns(nel), dim))
            A = one(Tensor{2, dim}) + 0.3 * Tensor{2, dim}((i, j) -> i < j ? 1.0 : 0.0)
            transform_coordinates!(grid, x -> A ⋅ x + 0.1 * ones(Vec{dim}))
            return grid
        end
        # Non-affine node perturbation (for quads/hexes the cell mappings are then
        # genuinely non-affine; triangles/tets get varying affine maps).
        function perturbed_grid(CT, nel)
            dim = Ferrite.getrefdim(CT)
            grid = generate_grid(CT, ntuple(Returns(nel), dim))
            transform_coordinates!(grid, x -> x + 0.1 * Vec(ntuple(i -> sin(π * x[mod1(i + 1, dim)]) / 2, dim)))
            return grid
        end

        function interpolation_error(dh, ip, f; qr_order = 4)
            CT = Ferrite.getcelltype(Ferrite.get_grid(dh))
            cv = CellValues(QuadratureRule{Ferrite.getrefshape(ip)}(qr_order), ip, Ferrite.geometric_interpolation(CT))
            a = zeros(ndofs(dh))
            apply_analytical!(a, dh, :u, f)
            err = 0.0
            for cc in CellIterator(dh)
                reinit!(cv, cc)
                ae = a[celldofs(cc)]
                for qp in 1:getnquadpoints(cv)
                    x = spatial_coordinate(cv, qp, getcoordinates(cc))
                    err += norm(function_value(cv, qp, ae) - f(x))^2 * getdetJdV(cv, qp)
                end
            end
            return sqrt(err)
        end

        rot2(x::Vec{2}) = Vec(-x[2], x[1])
        const2_lin(x::Vec{2}) = Vec(1.0, 2.0) + 0.5 * x
        const2_rot(x::Vec{2}) = Vec(1.0, 2.0) + 0.5 * rot2(x)
        full_lin(x::Vec{2}) = Vec(1.0 + 2 * x[1] - x[2], 2.0 - x[1] + 0.5 * x[2])
        const3_lin(x::Vec{3}) = Vec(1.0, 2.0, -1.0) + 0.5 * x
        smooth2(x::Vec{2}) = Vec(sin(x[1] + 0.5 * x[2]), cos(x[2]) * (1 + x[1]))
        smooth3(x::Vec{3}) = Vec(sin(x[1] + x[3]), cos(x[2]), x[3] * exp(x[1] / 2))
        smooth(x::Vec{2}) = smooth2(x)
        smooth(x::Vec{3}) = smooth3(x)

        # Interpolations with a function contained in the FE space on affine grids
        cases = [
            (RaviartThomas{RefTriangle, 1}(), Triangle, const2_lin),
            (RaviartThomas{RefTriangle, 2}(), Triangle, full_lin),
            (RaviartThomas{RefQuadrilateral, 1}(), Quadrilateral, const2_lin),
            (RaviartThomas{RefTetrahedron, 1}(), Tetrahedron, const3_lin),
            (RaviartThomas{RefHexahedron, 1}(), Hexahedron, const3_lin),
            (BrezziDouglasMarini{RefTriangle, 1}(), Triangle, full_lin),
            (Nedelec{RefTriangle, 1}(), Triangle, const2_rot),
            (Nedelec{RefTriangle, 2}(), Triangle, full_lin),
            (Nedelec{RefQuadrilateral, 1}(), Quadrilateral, const2_rot),
        ]

        @testset "Exact reproduction (affine): $ip" for (ip, CT, f) in cases
            dim = Ferrite.getrefdim(CT)
            grid = affine_grid(CT, dim == 2 ? 3 : 2)
            dh = close!(add!(DofHandler(grid), :u, ip))
            @test interpolation_error(dh, ip, f) < 1.0e-12
            # A higher quadrature order must not change exactness
            a = zeros(ndofs(dh))
            a_qr = zeros(ndofs(dh))
            apply_analytical!(a, dh, :u, f)
            apply_analytical!(a_qr, dh, :u, f; qr_order = 2 * Ferrite.getorder(ip) + 2)
            @test a_qr ≈ a atol = 1.0e-12
        end

        @testset "Shared facet dofs (non-affine): $ip" for (ip, CT, _) in cases
            dim = Ferrite.getrefdim(CT)
            grid = perturbed_grid(CT, 2)
            dh = close!(add!(DofHandler(grid), :u, ip))
            # Find two cells sharing a facet (the only shared dofs for these ips)
            AB = findfirst(
                ((A, B),) -> !isempty(intersect(celldofs(dh, A), celldofs(dh, B))),
                [(A, B) for A in 1:getncells(grid) for B in (A + 1):getncells(grid)]
            )
            A, B = [(A, B) for A in 1:getncells(grid) for B in (A + 1):getncells(grid)][AB]
            shared_dofs = intersect(celldofs(dh, A), celldofs(dh, B))
            @assert !isempty(shared_dofs)
            a_A = fill(NaN, ndofs(dh))
            a_B = fill(NaN, ndofs(dh))
            apply_analytical!(a_A, dh, :u, smooth, [A])
            apply_analytical!(a_B, dh, :u, smooth, [B])
            @test a_A[shared_dofs] ≈ a_B[shared_dofs]
            # Only dofs of the given cellset may be written
            @test all(isnan, a_A[setdiff(1:ndofs(dh), celldofs(dh, A))])
        end

        @testset "Curved geometry smoke test" begin
            # Quadratic geometric interpolation with non-affinely perturbed (mid)nodes
            grid = generate_grid(QuadraticTriangle, (2, 2))
            transform_coordinates!(grid, x -> x + 0.05 * Vec(sin(π * x[2]), sin(π * x[1])))
            for ip in (RaviartThomas{RefTriangle, 2}(), Nedelec{RefTriangle, 2}())
                dh = close!(add!(DofHandler(grid), :u, ip))
                shared_dofs = intersect(celldofs(dh, 1), celldofs(dh, 2))
                @assert !isempty(shared_dofs)
                a_A = fill(NaN, ndofs(dh))
                a_B = fill(NaN, ndofs(dh))
                apply_analytical!(a_A, dh, :u, smooth2, [1])
                apply_analytical!(a_B, dh, :u, smooth2, [2])
                @test a_A[shared_dofs] ≈ a_B[shared_dofs]
            end
        end

        @testset "ProjectedDirichlet equivalence: $ip" for (ip, CT, _) in cases
            Ferrite.getrefdim(CT) == 3 && continue # keep runtime down, 2D covers the kernels
            grid = perturbed_grid(CT, 2)
            dh = close!(add!(DofHandler(grid), :u, ip))
            facetset = union(getfacetset(grid, "left"), getfacetset(grid, "top"))
            f_bc = if Ferrite.conformity(ip) isa Ferrite.HdivConformity
                (x, t, n) -> smooth2(x) ⋅ n
            else
                (x, t, n) -> smooth2(x) × n
            end
            ch = close!(add!(ConstraintHandler(dh), ProjectedDirichlet(:u, facetset, f_bc)))
            update!(ch, 0.0)
            a_ch = zeros(ndofs(dh))
            apply!(a_ch, ch)
            a_an = zeros(ndofs(dh))
            apply_analytical!(a_an, dh, :u, smooth2)
            @test a_ch[ch.prescribed_dofs] ≈ a_an[ch.prescribed_dofs]
        end

        @testset "Convergence (non-affine)" begin
            for (ip, CT) in ((RaviartThomas{RefTriangle, 1}(), Triangle), (Nedelec{RefQuadrilateral, 1}(), Quadrilateral))
                errs = map((4, 8)) do nel
                    grid = perturbed_grid(CT, nel)
                    dh = close!(add!(DofHandler(grid), :u, ip))
                    return interpolation_error(dh, ip, smooth2)
                end
                # First order interpolations: L2 error = O(h)
                @test 1.5 < errs[1] / errs[2] < 3.0
            end
        end

        @testset "Mixed DofHandler and subdomains: $ip" for ip in (RaviartThomas{RefTriangle, 2}(), Nedelec{RefTriangle, 2}())
            grid = perturbed_grid(Triangle, 2)
            dh_single = close!(add!(DofHandler(grid), :u, ip))
            a_single = zeros(ndofs(dh_single))
            apply_analytical!(a_single, dh_single, :u, smooth2)
            # :u is the second field, so its dofs are offset in the cell dof vector
            dh_mixed = DofHandler(grid)
            add!(dh_mixed, :p, Lagrange{RefTriangle, 1}())
            add!(dh_mixed, :u, ip)
            close!(dh_mixed)
            a_mixed = fill(NaN, ndofs(dh_mixed))
            apply_analytical!(a_mixed, dh_mixed, :u, smooth2)
            for cellnr in 1:getncells(grid)
                dofs_single = celldofs(dh_single, cellnr)[dof_range(dh_single.subdofhandlers[1], :u)]
                dofs_mixed = celldofs(dh_mixed, cellnr)[dof_range(dh_mixed.subdofhandlers[1], :u)]
                @test a_mixed[dofs_mixed] ≈ a_single[dofs_single]
            end
            # :p dofs untouched, and the nodal path accepts (and ignores) qr_order
            pdofs = setdiff(1:ndofs(dh_mixed), reduce(vcat, [celldofs(dh_mixed, i)[dof_range(dh_mixed.subdofhandlers[1], :u)] for i in 1:getncells(grid)]))
            @test all(isnan, a_mixed[pdofs])
            apply_analytical!(a_mixed, dh_mixed, :p, x -> norm(x)^2; qr_order = 4)
            @test !any(isnan, a_mixed)

            # Field only on a subdomain
            set_A = Set(1:(getncells(grid) ÷ 2))
            dh_sub = DofHandler(grid)
            sdh_A = SubDofHandler(dh_sub, set_A)
            add!(sdh_A, :u, ip)
            close!(dh_sub)
            a_sub = zeros(ndofs(dh_sub))
            apply_analytical!(a_sub, dh_sub, :u, smooth2)
            for cellnr in set_A
                dofs_single = celldofs(dh_single, cellnr)[dof_range(dh_single.subdofhandlers[1], :u)]
                @test a_sub[celldofs(dh_sub, cellnr)] ≈ a_single[dofs_single]
            end
        end

        @testset "Eltype generality: $ip" for ip in (
                RaviartThomas{RefTriangle, 1}(),   # 1x1 facet systems
                RaviartThomas{RefTriangle, 2}(),   # 2x2 facet + 2x2 interior systems
                BrezziDouglasMarini{RefTriangle, 1}(),
                Nedelec{RefTriangle, 2}(),
            )
            grid = perturbed_grid(Triangle, 2)
            dh = close!(add!(DofHandler(grid), :u, ip))
            a64 = zeros(Float64, ndofs(dh))
            a32 = zeros(Float32, ndofs(dh))
            apply_analytical!(a64, dh, :u, smooth2)
            apply_analytical!(a32, dh, :u, smooth2)
            @test a32 ≈ a64 rtol = 1.0e-4
        end

        @testset "Non-nodal exceptions" begin
            grid3 = generate_grid(Tetrahedron, (1, 1, 1))
            dh3 = close!(add!(DofHandler(grid3), :u, Nedelec{RefTetrahedron, 1}()))
            a3 = rand(ndofs(dh3))
            a3_0 = copy(a3)
            @test_throws ArgumentError apply_analytical!(a3, dh3, :u, x -> Vec(0.0, 0.0, 0.0))
            @test a3 == a3_0 # `a` must not be partially modified when erroring
            grid2 = generate_grid(Triangle, (1, 1))
            dh2 = close!(add!(DofHandler(grid2), :u, RaviartThomas{RefTriangle, 1}()))
            @test_throws ErrorException apply_analytical!(zeros(ndofs(dh2)), dh2, :u, x -> 0.0) # Should be f(x)::Vec{2}
        end
    end
end
