# Imports for parallel (isolated) test execution:
using LinearAlgebra
using HCubature: hcubature, hquadrature
include(joinpath(@__DIR__, "test_utils.jl"))
include(joinpath(@__DIR__, "interpolation_test_utils.jl"))

using Ferrite: reference_shape_value

@testset "H(curl) and H(div)" begin
    Hcurl_interpolations = [
        Nedelec{RefTriangle, 1}(), Nedelec{RefTriangle, 2}(), Nedelec{RefQuadrilateral, 1}(),
        Nedelec{RefTetrahedron, 1}(), Nedelec{RefHexahedron, 1}(),
    ]
    Hdiv_interpolations = [
        RaviartThomas{RefTriangle, 1}(), RaviartThomas{RefTriangle, 2}(), RaviartThomas{RefQuadrilateral, 1}(),
        RaviartThomas{RefTetrahedron, 1}(), RaviartThomas{RefHexahedron, 1}(),
        BrezziDouglasMarini{RefTriangle, 1}(),
    ]

    # These reference moments define the functionals that an interpolation should fulfill
    ## Raviart-Thomas on RefTriangle
    reference_edge_moment(::RaviartThomas{RefTriangle, 1}, edge_shape_nr, s) = 1
    reference_edge_moment(::RaviartThomas{RefTriangle, 2}, edge_shape_nr, s) = edge_shape_nr == 1 ? (1 - s) : s
    reference_face_moment(::RaviartThomas{RefTriangle, 2}, face_shape_nr, s1, s2) = face_shape_nr == 1 ? Vec(1, 0) : Vec(0, 1)

    ## Raviart-Thomas on RefQuadrilateral
    reference_edge_moment(::RaviartThomas{RefQuadrilateral, 1}, edge_shape_nr, s) = 1

    ## Raviart-Thomas on RefTetrahedron
    reference_face_moment(::RaviartThomas{RefTetrahedron, 1}, face_shape_nr, s1, s2) = 1

    ## Raviart-Thomas on RefHexahedron
    reference_face_moment(::RaviartThomas{RefHexahedron, 1}, face_shape_nr, s1, s2) = 1

    ## Brezzi-Douglas-Marini on RefTriangle
    reference_edge_moment(::BrezziDouglasMarini{RefTriangle, 1}, edge_shape_nr, s) = edge_shape_nr == 1 ? (1 - s) : s

    ## Nedelec on RefTriangle
    reference_edge_moment(::Nedelec{RefTriangle, 1}, edge_shape_nr, s) = 1
    reference_edge_moment(::Nedelec{RefTriangle, 2}, edge_shape_nr, s) = edge_shape_nr == 1 ? (1 - s) : s
    reference_face_moment(::Nedelec{RefTriangle, 2}, face_shape_nr, s1, s2) = face_shape_nr == 1 ? Vec(1, 0) : Vec(0, 1)

    ## Nedelec on RefQuadrilateral
    reference_edge_moment(::Nedelec{RefQuadrilateral, 1}, edge_shape_nr, s) = 1

    ## Nedelec on RefTetrahedron
    reference_edge_moment(::Nedelec{RefTetrahedron, 1}, edge_shape_nr, s) = 1

    ## Nedelec on RefHexahedron
    reference_edge_moment(::Nedelec{RefHexahedron, 1}, edge_shape_nr, s) = 1

    """
        integrate_edge(f)

    Integrate f(s) on the unit line, s ∈ [0, 1]
    """
    function integrate_edge(f::Function)
        val, _ = hquadrature(f, 0, 1; atol = 1.0e-8)
        return val
    end

    """
        integrate_face(f)

    Integrate f(s1, s2) on the unit square, (s1,s2) ∈ [0, 1] × [0, 1]
    """
    function integrate_face(f::Function)
        val, _ = hcubature(z -> f(z[1], z[2]), (0, 0), (1, 1); atol = 1.0e-8)
        return val
    end

    parameterize_edge(edge_coords, s) = edge_coords[1] + (edge_coords[2] - edge_coords[1]) * s

    function parameterize_face(face_coords, s1, s2)
        ξ0 = face_coords[1]
        v1 = face_coords[2] - ξ0
        v2 = face_coords[end] - ξ0
        if length(face_coords) == 3 # Triangle
            return ξ0 + s1 * (1 - s2 / 2) * v1 + s2 * (1 - s1 / 2) * v2
        elseif length(face_coords) == 4 # Quadrilateral
            return ξ0 + s1 * v1 + s2 * v2
        else
            throw(ArgumentError("length(face_coords) must be 3 or 4"))
        end
    end

    # parameterize_volume(volume_coords, s1, s2, s3)
    # TODO parameterization of volume (Tetrahedron, Hexahedron, Prism, Pyramid)

    edge_weight(ξ::F, s) where {F <: Function} = norm(gradient(ξ, s))
    face_weight(ξ::F, s1, s2) where {F <: Function} = norm(gradient(s -> ξ(s, s2), s1) × gradient(s -> ξ(s1, s), s2))

    function test_interpolation_functionals(ip::Interpolation)
        @testset "functionals $ip" begin
            test_interpolation_functionals(Ferrite.conformity(ip), Val(Ferrite.getrefdim(ip)), ip)
        end
    end

    function test_interpolation_functionals(::Ferrite.HdivConformity, ::Val{2}, ip::Interpolation)
        RefShape = getrefshape(ip)
        ipg = Lagrange{RefShape, 1}()

        # Test functionals associated with the edges
        for edge_nr in 1:Ferrite.nedges(RefShape)
            edge_coords = getindex.((Ferrite.reference_coordinates(ipg),), Ferrite.reference_edges(RefShape)[edge_nr])
            dof_inds = Ferrite.edgedof_interior_indices(ip)[edge_nr]

            ξ(s) = parameterize_edge(edge_coords, s)
            normal = reference_normals(RefShape)[edge_nr]
            for (edge_shape_nr, shape_nr) in pairs(dof_inds)
                f(s) = reference_edge_moment(ip, edge_shape_nr, s) * (reference_shape_value(ip, ξ(s), shape_nr) ⋅ normal) * edge_weight(ξ, s)
                @test integrate_edge(f) ≈ 1
                # Test that the functional is zero for the other shape functions
                for other_shape_nr in 1:getnbasefunctions(ip)
                    other_shape_nr == shape_nr && continue
                    g(s) = reference_edge_moment(ip, edge_shape_nr, s) * (reference_shape_value(ip, ξ(s), other_shape_nr) ⋅ normal) * edge_weight(ξ, s)
                    @test integrate_edge(g) + 1 ≈ 1 # integrate_edge(g) ≈ 0
                end
            end

            # Test that normal components of other shape functions are zero on this edge
            # Stronger requirement than functionals being zero.
            for shape_nr in 1:getnbasefunctions(ip)
                shape_nr in dof_inds && continue
                for s in range(0, 1, 5)
                    @test abs(reference_shape_value(ip, ξ(s), shape_nr) ⋅ normal) < 1.0e-10
                end
            end
        end

        # Test functionals associated with the faces
        for face_nr in 1:Ferrite.nfaces(RefShape)
            face_coords = getindex.((Ferrite.reference_coordinates(ipg),), Ferrite.reference_faces(RefShape)[face_nr])
            dof_inds = Ferrite.facedof_interior_indices(ip)[face_nr]

            ξ(s1, s2) = parameterize_face(face_coords, s1, s2)
            for (face_shape_nr, shape_nr) in pairs(dof_inds)
                f(s1, s2) = reference_face_moment(ip, face_shape_nr, s1, s2) ⋅ reference_shape_value(ip, ξ(s1, s2), shape_nr) * face_weight(ξ, s1, s2)
                @test integrate_face(f) ≈ 1

                # Test that the functional is zero for the other shape functions
                for other_shape_nr in 1:getnbasefunctions(ip)
                    other_shape_nr == shape_nr && continue
                    g(s1, s2) = reference_face_moment(ip, face_shape_nr, s1, s2) ⋅ reference_shape_value(ip, ξ(s1, s2), other_shape_nr) * face_weight(ξ, s1, s2)
                    @test integrate_face(g) + 1 ≈ 1 # integrate_edge(g) ≈ 0
                end
            end
        end
    end

    # 3D, H(div)
    function test_interpolation_functionals(::Ferrite.HdivConformity, ::Val{3}, ip::Interpolation)
        RefShape = getrefshape(ip)
        ipg = Lagrange{RefShape, 1}()

        # Test functionals associated with the faces
        for face_nr in 1:Ferrite.nfaces(RefShape)
            face_coords = getindex.((Ferrite.reference_coordinates(ipg),), Ferrite.reference_faces(RefShape)[face_nr])
            normal = reference_normals(RefShape)[face_nr]
            dof_inds = Ferrite.facedof_interior_indices(ip)[face_nr]

            ξ(s1, s2) = parameterize_face(face_coords, s1, s2)
            for (face_shape_nr, shape_nr) in pairs(dof_inds)
                f(s1, s2) = reference_face_moment(ip, face_shape_nr, s1, s2) * (reference_shape_value(ip, ξ(s1, s2), shape_nr) ⋅ normal) * face_weight(ξ, s1, s2)
                @test integrate_face(f) ≈ 1

                # Test that the functional is zero for the other shape functions
                for other_shape_nr in 1:getnbasefunctions(ip)
                    other_shape_nr == shape_nr && continue
                    g(s1, s2) = reference_face_moment(ip, face_shape_nr, s1, s2) * (reference_shape_value(ip, ξ(s1, s2), other_shape_nr) ⋅ normal) * face_weight(ξ, s1, s2)
                    @test integrate_face(g) + 1 ≈ 1 # integrate_edge(g) ≈ 0
                end
            end

            # Test that normal components of other shape functions are zero on this edge
            # Stronger requirement than functionals being zero.
            for shape_nr in 1:getnbasefunctions(ip)
                shape_nr in dof_inds && continue
                for s1 in range(0, 1, 5), s2 in range(0, 1, 5)
                    @test abs(reference_shape_value(ip, ξ(s1, s2), shape_nr) ⋅ normal) < 1.0e-10
                end
            end
        end

        # Test functionals associated with the volume
        @assert length(Ferrite.volumedof_interior_indices(ip)) == 0 # Should be supported when testing `ip`s with those.
    end

    # H(curl)
    function test_interpolation_functionals(::Ferrite.HcurlConformity, ::Union{Val{2}, Val{3}}, ip::Interpolation)
        RefShape = getrefshape(ip)
        ipg = Lagrange{RefShape, 1}()

        # Test functionals associated with the edges
        for edge_nr in 1:Ferrite.nedges(RefShape)
            edge_coords = getindex.((Ferrite.reference_coordinates(ipg),), Ferrite.reference_edges(RefShape)[edge_nr])
            tangent = normalize(edge_coords[2] - edge_coords[1])
            dof_inds = Ferrite.edgedof_interior_indices(ip)[edge_nr]

            ξ(s) = parameterize_edge(edge_coords, s)
            for (edge_shape_nr, shape_nr) in pairs(dof_inds)
                f(s) = reference_edge_moment(ip, edge_shape_nr, s) * (reference_shape_value(ip, ξ(s), shape_nr) ⋅ tangent) * edge_weight(ξ, s)
                @test integrate_edge(f) ≈ 1
            end

            for shape_nr in 1:getnbasefunctions(ip)
                shape_nr in dof_inds && continue # Already tested
                for s in range(0, 1, 5)
                    @test abs(reference_shape_value(ip, ξ(s), shape_nr) ⋅ tangent) < 1.0e-10
                end
            end
        end

        # Test functionals associated with the faces
        for face_nr in 1:Ferrite.nfaces(RefShape)
            face_coords = getindex.((Ferrite.reference_coordinates(ipg),), Ferrite.reference_faces(RefShape)[face_nr])
            dof_inds = Ferrite.facedof_interior_indices(ip)[face_nr]

            ξ(s1, s2) = parameterize_face(face_coords, s1, s2)
            for (face_shape_nr, shape_nr) in pairs(dof_inds)
                f(s1, s2) = reference_face_moment(ip, face_shape_nr, s1, s2) ⋅ reference_shape_value(ip, ξ(s1, s2), shape_nr) * face_weight(ξ, s1, s2)
                @test integrate_face(f) ≈ 1
            end
        end

        # Test functionals associated with volume
        @assert length(Ferrite.volumedof_interior_indices(ip)) == 0 # Test not supported yet, but needs to be if introduced ip with volumedofs.
    end

    @testset "Interpolation properties" begin
        test_interpolation_properties.(Hcurl_interpolations)
        test_interpolation_properties.(Hdiv_interpolations)
    end
    @testset "Interpolation functionals" begin
        test_interpolation_functionals.(Hcurl_interpolations)
        test_interpolation_functionals.(Hdiv_interpolations)
    end

    # Test boundary conditions (ProjectedDirichlet)
    # Depending on the interpolation, we have different polynomials orders on the facets
    _facet_poly_order(ip::Nedelec) = Ferrite.getorder(ip) - 1
    _facet_poly_order(ip::RaviartThomas) = Ferrite.getorder(ip) - 1
    _facet_poly_order(ip::BrezziDouglasMarini) = Ferrite.getorder(ip)
    # Based on this order, p_facet, we expect that we should fulfill different criteria.
    # * If we prescribe a polynomial function to ProjectedDirichlet with a lower or equal order
    #   than p_facet, we expect that the interpolation should match the provided function pointwise.
    # * If we prescribe a polynomial function with higher order, but lower order than what the quadrature
    #   rule in ProjectedDirichlet can integrate exactly, we expect that the integral over the boundary are equal.
    # * If we prescribe a polynomial one order lower than we can integrate exactly, and p_facet ≥ 1,
    #   we expect that the integrated linear moment equation, ∫ x f(x) dx, is integrated exactly
    # The following tests check those properties for the H(div) and H(curl) interpolations.

    cell_type(::Type{RefLine}) = Line
    cell_type(::Type{RefTriangle}) = Triangle
    cell_type(::Type{RefQuadrilateral}) = Quadrilateral
    cell_type(::Type{RefTetrahedron}) = Tetrahedron
    cell_type(::Type{RefHexahedron}) = Hexahedron
    function _setup_dh_fv_for_bc_test(ip::Interpolation; nel = 3, qr_order = 6)
        RefShape = Ferrite.getrefshape(ip)
        CT = cell_type(RefShape)
        dim = Ferrite.getrefdim(CT) # dim=sdim=vdim
        grid = generate_grid(CT, ntuple(Returns(nel), dim), -0.25 * ones(Vec{dim}), 0.2 * ones(Vec{dim}))
        transform_coordinates!(grid, x -> x + 0.25 * Vec(ntuple(i -> abs(x[dim - i + 1])^(1 + i / dim), dim)))
        qr = FacetQuadratureRule{RefShape}(qr_order)
        fv = FacetValues(qr, ip, geometric_interpolation(CT))
        dh = close!(add!(DofHandler(grid), :u, ip))
        return dh, (fv,)
    end
    function _setup_dh_fv_for_bc_test(ips::Tuple{Interpolation{<:RefTriangle}, Interpolation{<:RefQuadrilateral}}; nel = 3, qr_order = 6)
        dim = 2
        trigrid = generate_grid(Triangle, (3, 3), -0.25 * ones(Vec{dim}), 0.2 * ones(Vec{dim}))
        #trigrid = generate_grid(Triangle, (3, 3))
        mixgrid, nr_quad = grid_with_inserted_quad(trigrid, (3, 4); update_sets = false) # Quad on bottom facet.
        empty!(mixgrid.facetsets)
        addfacetset!(mixgrid, "bottom", x -> x[2] ≈ -0.25)
        transform_coordinates!(mixgrid, x -> x + 0.25 * Vec(ntuple(i -> abs(x[dim - i + 1])^(1 + i / dim), dim)))
        qr_tri = FacetQuadratureRule{RefTriangle}(qr_order)
        fv_tri = FacetValues(qr_tri, ips[1], geometric_interpolation(Triangle))
        qr_quad = FacetQuadratureRule{RefQuadrilateral}(qr_order)
        fv_quad = FacetValues(qr_quad, ips[2], geometric_interpolation(Quadrilateral))
        dh = DofHandler(mixgrid)
        sdh_tri = SubDofHandler(dh, setdiff(1:getncells(mixgrid), Set(nr_quad)))
        add!(sdh_tri, :u, ips[1])
        sdh_quad = SubDofHandler(dh, Set(nr_quad))
        add!(sdh_quad, :u, ips[2])
        close!(dh)
        return dh, (fv_tri, fv_quad)
    end

    function test_bc_integral(
            f_bc::Function, check_fun::Function, moment_fun::Function, dh::DofHandler, facetset, fvs::Tuple;
            tol = 1.0e-6, custom_qr_order = nothing, pointwise_check, kwargs...
        )
        grid = Ferrite.get_grid(dh)
        dbc = if custom_qr_order === nothing
            ProjectedDirichlet(:u, facetset, f_bc)
        else
            ProjectedDirichlet(:u, facetset, f_bc; qr_order = custom_qr_order)
        end
        ch = close!(add!(ConstraintHandler(dh), dbc))
        a = zeros(ndofs(dh))
        apply!(a, ch)
        fv1 = first(fvs)
        test_val = zero(f_bc(get_node_coordinate(grid, 1), 0.0, getnormal(fv1, 1)))
        check_val = zero(check_fun(zero(Ferrite.shape_value_type(fv1)), getnormal(fv1, 1)))
        @assert typeof(test_val) === typeof(check_val)
        for (sdh, fv) in zip(dh.subdofhandlers, fvs)
            tv, cv = test_bc_integral(f_bc, check_fun, moment_fun, sdh, a, facetset, fv; pointwise_check, kwargs...)
            test_val += tv
            check_val += cv
        end
        @test norm(test_val - check_val) < tol
    end

    function test_bc_integral(
            f_bc::Function, check_fun::Function, moment_fun::Function, sdh::SubDofHandler, a, facetset, fv::FacetValues;
            pointwise_check, patchwise_check = true
        )
        grid = sdh.dh.grid
        test_val = zero(f_bc(get_node_coordinate(grid, 1), 0.0, getnormal(fv, 1)))
        check_val = zero(check_fun(zero(Ferrite.shape_value_type(fv)), getnormal(fv, 1)))
        @assert typeof(test_val) === typeof(check_val)
        for (cellidx, facetidx) in facetset
            cellidx ∈ sdh.cellset || continue
            cell_coords = getcoordinates(grid, cellidx)
            reinit!(fv, getcells(grid, cellidx), cell_coords, facetidx)
            ae = a[celldofs(sdh, cellidx)]
            for q_point in 1:getnquadpoints(fv)
                dΓ = getdetJdV(fv, q_point)
                u = function_value(fv, q_point, ae)
                n = getnormal(fv, q_point)
                x = spatial_coordinate(fv, q_point, cell_coords)
                check_fun_val = check_fun(u, n)
                bc_fun_val = f_bc(x, 0.0, n)
                check_val += moment_fun(x) * check_fun_val * dΓ
                test_val += moment_fun(x) * bc_fun_val * dΓ
                if pointwise_check
                    @test check_fun_val ≈ bc_fun_val
                end
            end
            if patchwise_check
                @test check_val ≈ test_val
            end
        end
        return test_val, check_val
    end

    @testset "H(div) BC" begin
        for ip in Hdiv_interpolations
            @testset "$ip" begin
                dh, fv = _setup_dh_fv_for_bc_test(ip)
                linear_x1(x, _, _) = x[1]
                quadratic_x1(x, _, _) = (x[1] - 0.3)^2
                nonlinear(x, _, _) = 100 * (x[1] - 0.3)^4
                funs = [
                    (Returns(0.0), true),
                    (Returns(rand()), true),
                    (linear_x1, _facet_poly_order(ip) ≥ 1),
                    (quadratic_x1, _facet_poly_order(ip) ≥ 2),
                    (nonlinear, false), # Only test integral quantities using high-order integration.
                ]
                for (f_bc, pointwise_check) in funs
                    @testset "f_bc = $f_bc" begin
                        for facetset in values(dh.grid.facetsets)
                            custom_qr_order = f_bc === nonlinear ? 5 : nothing
                            test_bc_integral(f_bc, ⋅, Returns(1), dh, facetset, fv; pointwise_check, custom_qr_order)
                            if _facet_poly_order(ip) ≥ 1 && f_bc !== nonlinear # Linear facet polynomial should pass moment test.
                                test_bc_integral(f_bc, ⋅, x -> x[1], dh, facetset, fv; pointwise_check = false)
                            end
                        end
                    end
                end
            end
        end
    end

    @testset "H(curl) BC" begin
        ips = Any[Hcurl_interpolations...]
        push!(ips, (Nedelec{RefTriangle, 1}(), Nedelec{RefQuadrilateral, 1}())) # Mixed case
        for interp in ips
            ip = isa(interp, Tuple) ? interp[1] : interp
            dim = Ferrite.getrefdim(getrefshape(ip))
            dim == 3 && continue # TODO: Not yet implemented
            @testset "$interp" begin
                dh, fvs = _setup_dh_fv_for_bc_test(interp)
                v3 = rand()
                linear_x1(x, _, _) = x[1] * Vec((0.0, 0.0, v3))
                quadratic_x1(x, _, _) = (x[1] - 0.3)^2 * Vec((0.0, 0.0, v3))
                nonlinear(x, _, _) = Vec((0.0, 0.0, v3)) * (100 * (x[1] + 0.3)^4)
                funs = [
                    (Returns(zero(Vec{3})), true),
                    (Returns(Vec((0.0, 0.0, rand()))), true),
                    (linear_x1, _facet_poly_order(ip) ≥ 1),
                    (quadratic_x1, _facet_poly_order(ip) ≥ 2),
                    (nonlinear, false),
                ]
                for (f_bc, pointwise_check) in funs
                    @testset "f_bc = $f_bc" begin
                        for facetset in values(dh.grid.facetsets)
                            custom_qr_order = f_bc === nonlinear ? 5 : nothing
                            test_bc_integral(f_bc, ×, Returns(1.0), dh, facetset, fvs; pointwise_check, custom_qr_order)
                            if _facet_poly_order(ip) ≥ 1  && f_bc !== nonlinear
                                test_bc_integral(f_bc, ×, x -> x[1], dh, facetset, fvs; pointwise_check = false)
                            end
                        end
                    end
                end
            end
        end
    end

    @testset "solve_small_spd!" begin
        # The 1x1/2x2 fast paths must be scale-safe: forming products of the
        # matrix entries (e.g. Cramer's rule) would overflow for these scales.
        for T in (Float32, Float64)
            overflow_scale = T == Float32 ? T(1.0e20) : 1.0e160
            for scale in (T(1) / overflow_scale, T(1), overflow_scale)
                K2 = T[2 1 / 2; 1 / 2 1] .* scale
                x_exact = T[1, -2]
                b = K2 * x_exact
                x = zeros(T, 2)
                Ferrite.solve_small_spd!(x, copy(K2), b)
                @test x ≈ x_exact rtol = sqrt(eps(T))
                K1 = fill(scale, 1, 1)
                Ferrite.solve_small_spd!(view(x, 1:1), K1, T[3scale])
                @test x[1] ≈ 3 rtol = sqrt(eps(T))
            end
        end
        # Generic (Cholesky) path against `\`
        M = [2.0 0.5 0.1; 0.5 1.0 0.2; 0.1 0.2 3.0]
        b3 = [1.0, -2.0, 0.5]
        x3 = zeros(3)
        Ferrite.solve_small_spd!(x3, copy(M), copy(b3))
        @test x3 ≈ M \ b3
    end

    @testset "ProjectedDirichlet in mixed DofHandler" begin
        # Regression test: the constrained field is not the first field in the
        # DofHandler, so its facet dofs are offset in the cell dof vector.
        for ip in (RaviartThomas{RefTriangle, 2}(), Nedelec{RefTriangle, 2}())
            grid = generate_grid(Triangle, (2, 2))
            transform_coordinates!(grid, x -> x + 0.1 * Vec(abs(x[2])^2, abs(x[1])^2))
            facetset = getfacetset(grid, "right")
            f_bc = if Ferrite.conformity(ip) isa Ferrite.HdivConformity
                (x, t, n) -> x[1] + 2 * x[2]
            else
                (x, t, n) -> Vec(0.0, 0.0, x[1] + 2 * x[2])
            end
            function solve_with_fields(fields...)
                dh = DofHandler(grid)
                for (name, field_ip) in fields
                    add!(dh, name, field_ip)
                end
                close!(dh)
                ch = close!(add!(ConstraintHandler(dh), ProjectedDirichlet(:u, facetset, f_bc)))
                a = zeros(ndofs(dh))
                apply!(a, ch)
                return dh, a
            end
            dh_single, a_single = solve_with_fields((:u, ip))
            dh_mixed, a_mixed = solve_with_fields((:p, Lagrange{RefTriangle, 1}()), (:u, ip))
            for (cellidx, _) in facetset
                dofs_single = celldofs(dh_single, cellidx)[dof_range(dh_single.subdofhandlers[1], :u)]
                dofs_mixed = celldofs(dh_mixed, cellidx)[dof_range(dh_mixed.subdofhandlers[1], :u)]
                @test a_mixed[dofs_mixed] ≈ a_single[dofs_single]
            end
        end
    end

    @testset "ProjectedDirichlet error path" begin
        dh_H1, _ = _setup_dh_fv_for_bc_test(Lagrange{RefTriangle, 1}()^2; nel = 1, qr_order = 1)
        dh_L2, _ = _setup_dh_fv_for_bc_test(DiscontinuousLagrange{RefTriangle, 1}()^2; nel = 1, qr_order = 1)
        dh_Hcurl_3d, _ = _setup_dh_fv_for_bc_test(Nedelec{RefTetrahedron, 1}(); nel = 1, qr_order = 1)
        for dh in (dh_H1, dh_L2, dh_Hcurl_3d)
            dbc = ProjectedDirichlet(:u, Set([FacetIndex(1, 1)]), Returns(zero(Vec{2})))
            ch = add!(ConstraintHandler(dh), dbc)
            @test_throws "ProjectedDirichlet is not implemented for" close!(ch)
        end
    end

end # testset
