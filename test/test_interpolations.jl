
@testset "interpolations" begin

@testset "Value Type $value_type" for value_type in (Float32, Float64)
    @testset "Correctness of $interpolation" for interpolation in (
                      Lagrange{RefLine, 1}(),
                      Lagrange{RefLine, 2}(),
                      Lagrange{RefQuadrilateral, 1}(),
                      Lagrange{RefQuadrilateral, 2}(),
                      Lagrange{RefQuadrilateral, 3}(),
                      Lagrange{RefTriangle, 1}(),
                      Lagrange{RefTriangle, 2}(),
                      Lagrange{RefTriangle, 3}(),
                      Lagrange{RefTriangle, 4}(),
                      Lagrange{RefTriangle, 5}(),
                      Lagrange{RefHexahedron, 1}(),
                      Lagrange{RefHexahedron, 2}(),
                      Serendipity{RefQuadrilateral, 2}(),
                      Serendipity{RefHexahedron, 2}(),
                      Lagrange{RefTetrahedron, 1}(),
                      Lagrange{RefTetrahedron, 2}(),
                      Lagrange{RefPrism, 1}(),
                      Lagrange{RefPrism, 2}(),
                      Lagrange{RefPyramid, 1}(),
                      Lagrange{RefPyramid, 2}(),
                      #
                      DiscontinuousLagrange{RefLine, 0}(),
                      DiscontinuousLagrange{RefQuadrilateral, 0}(),
                      DiscontinuousLagrange{RefHexahedron, 0}(),
                      DiscontinuousLagrange{RefTriangle, 0}(),
                      DiscontinuousLagrange{RefTetrahedron, 0}(),
                      DiscontinuousLagrange{RefLine, 1}(),
                      DiscontinuousLagrange{RefQuadrilateral, 1}(),
                      DiscontinuousLagrange{RefHexahedron, 1}(),
                      DiscontinuousLagrange{RefTriangle, 1}(),
                      DiscontinuousLagrange{RefTetrahedron, 1}(),
                      DiscontinuousLagrange{RefPrism, 1}(),
                      DiscontinuousLagrange{RefPyramid, 1}(),
                      #
                      BubbleEnrichedLagrange{RefTriangle, 1}(),
                      #
                      CrouzeixRaviart{RefTriangle, 1}(),
    )
        # Test of utility functions
        ref_dim = Ferrite.getdim(interpolation)
        ref_shape = Ferrite.getrefshape(interpolation)
        func_order = Ferrite.getorder(interpolation)
        @test typeof(interpolation) <: Interpolation{ref_shape,func_order}

        # Note that not every element formulation exists for every order and dimension.
        applicable(Ferrite.getlowerorder, interpolation) && @test typeof(Ferrite.getlowerorder(interpolation)) <: Interpolation{ref_shape,func_order-1}
        @testset "transform face points" begin
            # Test both center point and random points on the face
            ref_coord = Ferrite.reference_coordinates(Lagrange{ref_shape, 1}())
            for face in 1:nfaces(interpolation)
                face_nodes = Ferrite.reference_faces(ref_shape)[face]
                center_coord = [0.0 for _ in 1:ref_dim]
                rand_coord = [0.0 for _ in 1:ref_dim]
                rand_weights = rand(length(face_nodes))
                rand_weights /= sum(rand_weights)
                for (i, node) in pairs(face_nodes)
                    center_coord += ref_coord[node] / length(face_nodes)
                    rand_coord += rand_weights[i] .* ref_coord[node]
                end
                for point in (center_coord, rand_coord)
                    vec_point = Vec{ref_dim}(point)
                    cell_to_face = Ferrite.element_to_face_transformation(vec_point, ref_shape, face)
                    face_to_cell = Ferrite.face_to_element_transformation(cell_to_face, ref_shape, face)
                    @test vec_point ≈ face_to_cell
                end
            end
        end
        n_basefuncs = getnbasefunctions(interpolation)
        coords = Ferrite.reference_coordinates(interpolation)
        @test length(coords) == n_basefuncs
        f(x) = [shape_value(interpolation, Tensor{1, ref_dim}(x), i) for i in 1:n_basefuncs]

        #TODO prefer this test style after 1.6 is removed from CI
        # @testset let x = sample_random_point(ref_shape) # not compatible with Julia 1.6
        x = Vec{ref_dim,value_type}(sample_random_point(ref_shape))
        random_point_testset = @testset "Random point test" begin
            # Check gradient evaluation
            @test vec(ForwardDiff.jacobian(f, Array(x))') ≈
                reinterpret(value_type, [shape_gradient(interpolation, x, i) for i in 1:n_basefuncs])
            # Check partition of unity at random point.
            @test sum([shape_value(interpolation, x, i) for i in 1:n_basefuncs]) ≈ 1.0
            # Check if the important functions are consistent
            @test_throws ArgumentError shape_value(interpolation, x, n_basefuncs+1)
            # Idempotency test
            @test shape_value(interpolation, x, n_basefuncs) == shape_value(interpolation, x, n_basefuncs)
        end
        # Remove after 1.6 is removed from CI (see above)
        # Show coordinate in case failure (see issue #811)
        !isempty(random_point_testset.results) && println("^^^^^Random point test failed at $x for $interpolation !^^^^^")

        # Test whether we have for each entity corresponding dof indices (possibly empty)
        @test length(Ferrite.vertexdof_indices(interpolation)) == Ferrite.nvertices(interpolation)
        if ref_dim > 1
            @test length(Ferrite.facedof_indices(interpolation)) == Ferrite.nfaces(interpolation)
            @test length(Ferrite.facedof_interior_indices(interpolation)) == Ferrite.nfaces(interpolation)
        elseif ref_dim > 2
            @test length(Ferrite.edgedof_indices(interpolation)) == Ferrite.nedges(interpolation)
            @test length(Ferrite.edgedof_interior_indices(interpolation)) == Ferrite.nedges(interpolation)
        end
        # We have at least as many edge/face dofs as we have edge/face interior dofs
        if ref_dim > 1
            @test all(length.(Ferrite.facedof_interior_indices(interpolation)) .<= length.(Ferrite.facedof_indices(interpolation)))
        elseif ref_dim > 2
            @test all(length.(Ferrite.edgedof_interior_indices(interpolation)) .<= length.(Ferrite.edgedof_indices(interpolation)))
        end
        # The total number of dofs must match the number of base functions
        totaldofs = sum(length.(Ferrite.vertexdof_indices(interpolation));init=0)
        if ref_dim > 1
            totaldofs += sum(length.(Ferrite.facedof_interior_indices(interpolation));init=0)
        end
        if ref_dim > 2
            totaldofs += sum(length.(Ferrite.edgedof_interior_indices(interpolation));init=0)
        end
        totaldofs += length(Ferrite.celldof_interior_indices(interpolation))
        @test totaldofs == n_basefuncs

        # The dof indices are valid.
        @test all([all(0 .< i .<= n_basefuncs) for i ∈ Ferrite.vertexdof_indices(interpolation)])
        if ref_dim > 1
            @test all([all(0 .< i .<= n_basefuncs) for i ∈ Ferrite.facedof_indices(interpolation)])
            @test all([all(0 .< i .<= n_basefuncs) for i ∈ Ferrite.facedof_interior_indices(interpolation)])
        elseif ref_dim > 2
            @test all([all(0 .< i .<= n_basefuncs) for i ∈ Ferrite.edgedof_indices(interpolation)])
            @test all([all(0 .< i .<= n_basefuncs) for i ∈ Ferrite.edgedof_interior_indices(interpolation)])
        end
        @test all([all(0 .< i .<= n_basefuncs) for i ∈ Ferrite.celldof_interior_indices(interpolation)])

        # Check for evaluation type correctness of interpolation
        @testset "return type correctness dof $dof" for dof in 1:n_basefuncs
            @test (@inferred shape_value(interpolation, x, dof)) isa value_type
            @test (@inferred shape_gradient(interpolation, x, dof)) isa Vec{ref_dim, value_type}
        end

        # Check for dirac delta property of interpolation
        @testset "dirac delta property of dof $dof" for dof in 1:n_basefuncs
            for k in 1:n_basefuncs
                N_dof = shape_value(interpolation, coords[dof], k)
                if k == dof
                    @test N_dof ≈ 1.0
                else
                    factor = interpolation isa Lagrange{RefQuadrilateral, 3} ? 200 : 4
                    @test N_dof ≈ 0.0 atol = factor * eps(value_type)
                end
            end
        end

        # Test that facedof_indices(...) return in counter clockwise order (viewing from the outside)
        if interpolation isa Lagrange
            function __outward_normal(coords::Vector{<:Vec{1}}, nodes)
                n = coords[nodes[1]]
                return n / norm(n)
            end
            function __outward_normal(coords::Vector{<:Vec{2}}, nodes)
                p1 = coords[nodes[1]]
                p2 = coords[nodes[2]]
                n = Vec{2}((p2[2] - p1[2], - p2[1] + p1[1]))
                return n / norm(n)
            end
            function __outward_normal(coords::Vector{<:Vec{3}}, nodes)
                p1 = coords[nodes[1]]
                p2 = coords[nodes[2]]
                p3 = coords[nodes[3]]
                n = (p3 - p2) × (p1 - p2)
                return n / norm(n)
            end
            for (facenodes, normal) in zip(Ferrite.facedof_indices(interpolation), reference_normals(interpolation))
                @test __outward_normal(coords, facenodes) ≈ normal
            end
        end

        # regression for https://github.com/Ferrite-FEM/Ferrite.jl/issues/520
        interpolation_type = typeof(interpolation).name.wrapper
        if func_order > 1 && interpolation_type != Ferrite.Serendipity
            first_order = interpolation_type{ref_shape,1}()
            for (highorderface, firstorderface) in zip(Ferrite.facedof_indices(interpolation), Ferrite.facedof_indices(first_order))
                for (h_node, f_node) in zip(highorderface, firstorderface)
                    @test h_node == f_node
                end
            end
            if ref_dim > 2
                for (highorderedge, firstorderedge) in zip(Ferrite.edgedof_indices(interpolation), Ferrite.edgedof_indices(first_order))
                    for (h_node, f_node) in zip(highorderedge, firstorderedge)
                        @test h_node == f_node
                    end
                end
            end
        end

        # VectorizedInterpolation
        v_interpolation_1 = interpolation^2
        v_interpolation_2 = (d = 2; interpolation^d)
        @test getnbasefunctions(v_interpolation_1) ==
              getnbasefunctions(v_interpolation_2) ==
              getnbasefunctions(interpolation) * 2
        # pretty printing
        @test repr("text/plain", v_interpolation_1) == repr(v_interpolation_1.ip) * "^2"

        # Check for evaluation type correctness of vectorized interpolation
        v_interpolation_3 = interpolation^ref_dim
        @testset "vectorized case of return type correctness of dof $dof" for dof in 1:n_basefuncs
            @test @inferred(shape_value(v_interpolation_1, x, dof)) isa Vec{2, value_type}
            @test @inferred(shape_gradient(v_interpolation_3, x, dof)) isa Tensor{2, ref_dim, value_type}
        end
    end # correctness testset

    @test Ferrite.reference_coordinates(DiscontinuousLagrange{RefTriangle,0}()) ≈ [Vec{2,Float64}((1/3,1/3))]
    @test Ferrite.reference_coordinates(DiscontinuousLagrange{RefQuadrilateral,0}()) ≈ [Vec{2,Float64}((0,0))]
    @test Ferrite.reference_coordinates(DiscontinuousLagrange{RefTetrahedron,0}()) ≈ [Vec{3,Float64}((1/4,1/4,1/4))]
    @test Ferrite.reference_coordinates(DiscontinuousLagrange{RefHexahedron,0}()) ≈ [Vec{3,Float64}((0,0,0))]

    # Test discontinuous interpolations related functions
    d_ip = DiscontinuousLagrange{RefQuadrilateral,1}()
    d_ip_t = DiscontinuousLagrange{RefQuadrilateral,1}

    ip = Lagrange{RefQuadrilateral,1}()
    ip_t = Lagrange{RefQuadrilateral,1}

    @test Ferrite.is_discontinuous(ip) == false
    @test Ferrite.is_discontinuous(ip_t) == false
    @test Ferrite.is_discontinuous(d_ip) == true
    @test Ferrite.is_discontinuous(d_ip_t) == true
end

reference_cell(::Type{RefTriangle}) = Triangle((1,2,3))
reference_cell(::Type{RefQuadrilateral}) = Quadrilateral((1,2,3,4))
reference_cell(::Type{RefTetrahedron}) = Tetrahedron((1,2,3,4))
reference_cell(::Type{RefHexahedron}) = Hexahedron((ntuple(identity, 8)))

function line_integral(qr::QuadratureRule{RefLine, T}, ip, shape_nr, x0, Δx, L, v, f) where T
    s = zero(T)
    for (ξ1d, w) in zip(Ferrite.getpoints(qr), Ferrite.getweights(qr))
        ξ = x0 + (ξ1d[1]/2) * Δx
        s += (shape_value(ip, ξ, shape_nr) ⋅ v) * (w*L/2) * f((ξ1d[1]+1)/2)
    end
    return s
end

# Required properties of shape value Nⱼ of an edge-elements (Hcurl) on an edge with direction v, length L, and dofs ∈ 𝔇 
# 1) Unit property: ∑_{j∈𝔇} ∫(Nⱼ ⋅ v f(s) dS) = 1 
#    Where f(s) = 1 for linear interpolation and f(s)=1-s and f(s)=s for 2nd order interpolation (first and second shape function)
#    And s is the path parameter ∈[0,1] along the positive direction of the path. 
# 2) Zero along other edges: Nⱼ ⋅ v = 0 if j∉𝔇
@testset "Nedelec" begin
    lineqr = QuadratureRule{RefLine}(20)
    for ip in (Nedelec{2,RefTriangle,1}(), Nedelec{2,RefTriangle,2}(), Nedelec{3,RefTetrahedron,1}(), Nedelec{3,RefHexahedron,1}())
        cell = reference_cell(getrefshape(ip))
        edges = Ferrite.getdim(ip) == 2 ? Ferrite.faces(cell) : Ferrite.edges(cell)
        dofs = Ferrite.getdim(ip) == 2 ? Ferrite.facedof_interior_indices(ip) : Ferrite.edgedof_interior_indices(ip)
        x = Ferrite.reference_coordinates(Ferrite.default_interpolation(typeof(cell)))
        @testset "$(getrefshape(ip)), order=$(Ferrite.getorder(ip))" begin
            for (edge_nr, (i1, i2)) in enumerate(edges)
                Δx = x[i2]-x[i1]
                x0 = (x[i1]+x[i2])/2
                L = norm(Δx)
                v = Δx/L
                for (idof, shape_nr) in enumerate(dofs[edge_nr])
                    nedgedofs = length(dofs[edge_nr])
                    f(x) = nedgedofs == 1 ? 1.0 : (idof == 1 ? 1-x : x)
                    s = line_integral(lineqr, ip, shape_nr, x0, Δx, L, v, f)
                    @test s ≈ one(s)
                end
                for (j_edge, shape_nrs) in enumerate(dofs)
                    j_edge == edge_nr && continue
                    for shape_nr in shape_nrs
                        for ξ in (x[i1] + r*Δx for r in [0.0, rand(3)..., 1.0])
                            @test abs(shape_value(ip, ξ, shape_nr) ⋅ v) < eps()*100
                        end
                    end
                end
            end
        end
    end
end

tupleshift(t::NTuple{N}, shift::Int) where N = ntuple(i -> t[mod(i - 1 - shift, N) + 1], N)
#tupleshift(t::NTuple, shift::Int) = tuple(circshift(SVector(t), shift)...)
cell_permutations(cell::Quadrilateral)      = (Quadrilateral(tupleshift(cell.nodes, shift)) for shift in 0:3)
cell_permutations(cell::Triangle)           = (     Triangle(tupleshift(cell.nodes, shift)) for shift in 0:2)
cell_permutations(cell::QuadraticTriangle)  = (QuadraticTriangle((tupleshift(cell.nodes[1:3], shift)..., tupleshift(cell.nodes[4:6], shift)...)) for shift in 0:3)

function cell_permutations(cell::Hexahedron)
    idx = ( #Logic on refshape: Select 1st and 2nd vertex (must be neighbours)
    # The next follows to create inward vector with RHR, and then 4th is in same plane.
    # The last four must be the neighbours on the other plane to the first four (same order)
        (1,2,3,4,5,6,7,8), (1,4,8,5,2,3,7,6), (1,5,6,2,4,8,7,3),
        (2,1,5,6,3,4,8,7), (2,3,4,1,6,7,8,5), (2,6,7,3,1,5,8,4),
        (3,2,6,7,4,1,5,8), (3,4,1,2,7,8,5,6), (3,7,8,4,2,6,5,1),
        (4,1,2,3,8,5,6,7), (4,3,7,8,1,2,6,5), (4,8,5,1,3,7,6,1),
        (5,1,4,8,6,2,3,7), (5,6,2,1,8,7,3,4), (5,8,7,6,1,4,3,2),
        (6,2,1,5,7,3,4,8), (6,5,8,7,2,1,4,3), (6,7,3,2,5,8,4,1),
        (7,3,2,6,8,4,1,5), (7,6,5,8,3,2,1,4), (7,8,4,3,6,5,1,2),
        (8,4,3,7,5,1,2,6), (8,5,1,4,7,6,2,3), (8,7,6,5,4,3,2,1),
    )
    return (Hexahedron(ntuple(i -> cell.nodes[perm[i]], 8)) for perm in idx)
end

function cell_permutations(cell::Tetrahedron)
    idx = ( (1,2,3,4), (1,3,4,2), (1,4,2,3),
            (2,1,4,3), (2,3,1,4), (2,4,3,1),
            (3,1,2,4), (3,2,4,1), (3,4,1,2),
            (4,1,3,2), (4,3,2,1), (4,2,1,3))
    return (Tetrahedron(ntuple(i -> cell.nodes[perm[i]], 4)) for perm in idx)
end

@testset "Hcurl and Hdiv" begin
    include(joinpath(@__DIR__, "InterpolationTestUtils.jl"))
    import .InterpolationTestUtils as ITU
    nel = 3
    transformation_functions = Dict(
        Nedelec=>(v,n)-> v - n*(v⋅n),   # Hcurl (tangent continuity)
        RaviartThomas=>(v,n) -> v ⋅ n)  # Hdiv (normal continuity)
    for CT in (Triangle, QuadraticTriangle, Tetrahedron, Hexahedron)
        dim = Ferrite.getdim(CT)
        p1, p2 = (rand(Vec{dim}), ones(Vec{dim})+rand(Vec{dim}))
        grid = generate_grid(CT, ntuple(_->nel, dim), p1, p2)
        # Smoothly distort grid (to avoid spuriously badly deformed elements).
        # A distorted grid is important to properly test the geometry mapping
        # for 2nd order elements.
        transfun(x) = typeof(x)(i->sinpi(x[mod(i, length(x))+1]+i/3))/10
        transform_coordinates!(grid, x->(x + transfun(x)))
        cellnr = getncells(grid)÷2 + 1 # Should be a cell in the center
        basecell = getcells(grid, cellnr)
        RefShape = Ferrite.getrefshape(basecell)
        for order in (1, 2)
            for IPT in (Nedelec, RaviartThomas)
                dim == 3 && order > 1 && continue
                IPT == RaviartThomas && (dim == 3 || order > 1) && continue
                IPT == RaviartThomas && (RefShape == RefHexahedron) && continue
                transformation_function=transformation_functions[IPT]
                ip = IPT{dim,RefShape,order}()
                @testset "$CT, $ip" begin
                    for testcell in cell_permutations(basecell)
                        grid.cells[cellnr] = testcell
                        dh = DofHandler(grid)
                        add!(dh, :u, ip)
                        close!(dh)
                        for facenr in 1:nfaces(RefShape)
                            fi = FaceIndex(cellnr, facenr)
                            # Check continuity of tangential function value
                            ITU.test_continuity(dh, fi; transformation_function)
                        end
                        # Check gradient calculation 
                        ITU.test_gradient(dh, cellnr)
                    end
                end
            end
        end
    end
end

end # testset
