using Ferrite: reference_shape_value

@testset "Quadrature testing" begin
    ref_tet_vol(dim) = 1 / factorial(dim)
    ref_prism_vol() = 1 / 2
    ref_square_vol(dim) = 2^dim

    function integrate(qr::QuadratureRule, f::Function)
        I = 0.0
        for (w, x) in zip(Ferrite.getweights(qr), Ferrite.getpoints(qr))
            I += w * f(x)
        end
        return I
    end

    # Hypercube
    @testset "Exactness for integration on hypercube of $rulename" for (rulename, orderrange) in [
            (:legendre, 1:4),
            (:lobatto, 2:4),
        ]
        for (dim, shape) in ((1, RefLine), (2, RefQuadrilateral), (3, RefHexahedron))
            for order in orderrange
                f = (x, p) -> sum([x[i]^p for i in 1:length(x)])
                qr = QuadratureRule{shape}(rulename, order)
                @test integrate(qr, (x) -> f(x, 2 * order - 1)) ≈ 0.0 atol = 1.0e-14
                @test sum(qr.weights) ≈ ref_square_vol(dim)
                @test sum(Ferrite.getweights(qr)) ≈ ref_square_vol(dim)
            end
        end
    end
    @test_throws ArgumentError QuadratureRule{RefLine}(:einstein, 2)

    # Triangle
    # http://www.wolframalpha.com/input/?i=integrate+sqrt(x%2By)+from+x+%3D+0+to+1,+y+%3D+0+to+1-x
    @testset "Exactness for integration on triangles of $rulename" for (rulename, orderrange) in [
            (:dunavant, 1:8),
            (:gaussjacobi, 9:15),
        ]
        g = (x) -> sqrt(sum(x))
        dim = 2
        for order in orderrange
            qr = QuadratureRule{RefTriangle}(rulename, order)
            @test integrate(qr, g) ≈ 0.4 atol = 0.01 / order^2
            @test sum(qr.weights) ≈ ref_tet_vol(dim)
        end
    end
    @test_throws ArgumentError QuadratureRule{RefTriangle}(:einstein, 2)
    @test_throws ArgumentError QuadratureRule{RefTriangle}(0)

    # Tetrahedron
    # Test taken from Table 1 in dx.doi.org/10.1016/j.amc.2007.01.014
    # http://www.m-hikari.com/ijma/ijma-2011/ijma-1-4-2011/venkateshIJMA1-4-2011.pdf
    @testset "Exactness for integration on tetrahedra of $rulename" for (rulename, orderrange) in [
            (:jinyun, 1:3),
            (:keast_minimal, 1:5),
            (:keast_positive, 1:5),
        ]
        g = (x) -> sqrt(sum(x))
        dim = 3
        for order in orderrange
            qr = QuadratureRule{RefTetrahedron}(rulename, order)
            @test integrate(qr, g) ≈ 0.142857142857143 atol = 0.01 / order^2
            @test sum(qr.weights) ≈ ref_tet_vol(dim)
        end
    end
    @test_throws ArgumentError QuadratureRule{RefTetrahedron}(:einstein, 2)
    @test_throws ArgumentError QuadratureRule{RefTetrahedron}(0)

    # Wedge
    # ∫ √(x₁ + x₂) x₃² = 2 / 15
    @testset "Exactness for integration on prisms of $rulename" for (rulename, orderrange) in [
            (:polyquad, 1:10),
        ]
        g = (x) -> √(x[1] + x[2]) * x[3]^2
        for order in 1:10
            qr = QuadratureRule{RefPrism}(:polyquad, order)
            @test integrate(qr, g) ≈ 2 / 15 atol = 0.1 / order^2
            @test sum(qr.weights) ≈ ref_prism_vol()
        end
    end
    @test_throws ArgumentError QuadratureRule{RefPrism}(:einstein, 2)
    @test_throws ArgumentError QuadratureRule{RefPrism}(0)

    @testset "Generic quadrature rule properties for $ref_cell" for ref_cell in (
            Line,
            Quadrilateral,
            Triangle,
            Hexahedron,
            Tetrahedron,
            Wedge,
            Pyramid,
        )

        refshape = ref_cell.super.parameters[1]
        dim = refshape.super.parameters[1]

        dim > 1 && @testset "$refshape face-cell spatial coordinates" begin
            grid = generate_grid(ref_cell, ntuple(_ -> 3, dim))
            for cellid in 1:getncells(grid)
                cell = grid.cells[cellid]
                ccoords = getcoordinates(grid, cellid)
                Vec_t = Vec{dim, Float64}
                Vec_face_t = Vec{dim - 1, Float64}
                for lfaceid in Ferrite.nfacets(refshape)
                    facenodes = Ferrite.facets(cell)[lfaceid]
                    fcoords = zeros(Vec_t, length(facenodes))
                    for (i, nodeid) in enumerate(facenodes)
                        x = grid.nodes[nodeid].x
                        fcoords[i] = x
                    end
                    ipcell = Lagrange{refshape, 1}()
                    ipface = Lagrange{getfacetrefshape(refshape, lfaceid), 1}()

                    ξface = rand(Vec_face_t) / 4
                    ξcell = Ferrite.facet_to_element_transformation(ξface, refshape, lfaceid)

                    xface = zero(Vec_t)
                    for i in 1:getnbasefunctions(ipface)
                        xface += reference_shape_value(ipface, ξface, i) * fcoords[i]
                    end

                    xcell = zero(Vec_t)
                    for i in 1:getnbasefunctions(ipcell)
                        xcell += reference_shape_value(ipcell, ξcell, i) * ccoords[i]
                    end

                    @test xcell ≈ xface
                end
            end
        end

        @testset "$ref_cell unknown facet error path" begin
            for face in (-1, 0, 100)
                err = ArgumentError("unknown facet number")
                @test_throws err Ferrite.weighted_normal(Tensor{2, dim}(zeros(dim^2)), refshape, face)
                pt = Vec{dim - 1, Float64}(ntuple(i -> 0.0, dim - 1))
                @test_throws err Ferrite.facet_to_element_transformation(pt, refshape, face)
            end
        end

        @testset "Type checks for $refshape (T=$T)" for T in (Float32, Float64)
            qr = QuadratureRule{refshape}(T, 1)
            qrw = Ferrite.getweights(qr)
            qrp = Ferrite.getpoints(qr)
            @test qrw isa Vector
            @test qrp isa Vector
            @test eltype(qrw) === T
            @test eltype(eltype(qrp)) === T

            sqr = QuadratureRule{refshape}(
                SVector{length(qrw)}(qrw), SVector{length(qrp)}(qrp)
            )
            sqrw = Ferrite.getweights(sqr)
            sqrp = Ferrite.getpoints(sqr)
            @test sqrw isa SVector
            @test sqrp isa SVector
            @test eltype(sqrw) === T
            @test eltype(eltype(sqrp)) === T

            fqr = FacetQuadratureRule{refshape}(T, 1)
            for f in 1:nfacets(refshape)
                fqrw = Ferrite.getweights(fqr, f)
                fqrp = Ferrite.getpoints(fqr, f)
                @test fqrw isa Vector
                @test fqrp isa Vector
                @test eltype(fqrw) === T
                @test eltype(eltype(fqrp)) === T
            end

            function sqr_for_facet(fqr, f)
                fqrw = Ferrite.getweights(fqr, f)
                fqrp = Ferrite.getpoints(fqr, f)
                return QuadratureRule{refshape}(
                    SVector{length(qrw)}(fqrw),
                    SVector{length(qrp)}(fqrp),
                )
            end

            sfqr = FacetQuadratureRule(
                ntuple(f -> sqr_for_facet(fqr, f), nfacets(refshape))
            )
            for f in 1:nfacets(refshape)
                sfqrw = Ferrite.getweights(sfqr, f)
                sfqrp = Ferrite.getpoints(sfqr, f)
                @test sfqrw isa SVector
                @test sfqrp isa SVector
                @test eltype(sfqrw) === T
                @test eltype(eltype(sfqrp)) === T
            end

            sfqr2 = FacetQuadratureRule(
                [sqr_for_facet(fqr, f) for f in 1:nfacets(refshape)]
            )
            for f in 1:nfacets(refshape)
                sfqrw = Ferrite.getweights(sfqr2, f)
                sfqrp = Ferrite.getpoints(sfqr2, f)
                @test sfqrw isa SVector
                @test sfqrp isa SVector
                @test eltype(sfqrw) === T
                @test eltype(eltype(sfqrp)) === T
            end
        end
    end

    # Check explicitly if the defaults changed, as this might affect users negatively
    @testset "Volume defaults for $refshape" for (refshape, sym) in (
            (RefLine, :legendre),
            (RefQuadrilateral, :legendre),
            (RefHexahedron, :legendre),
            (RefTriangle, :dunavant),
            (RefTetrahedron, :keast_minimal),
            (RefPrism, :polyquad),
            (RefPyramid, :polyquad),
        )
        for order in 1:3
            qr = QuadratureRule{refshape}(sym, order)
            qr_default = QuadratureRule{refshape}(order)
            @test Ferrite.getweights(qr) == Ferrite.getweights(qr_default)
            @test Ferrite.getpoints(qr) == Ferrite.getpoints(qr_default)
        end
    end
    @testset "Facet defaults for $refshape" for (refshape, sym) in (
            # (RefLine, :legendre), # There is no choice for the rule on lines, as it only is a point eval
            (RefQuadrilateral, :legendre),
            (RefHexahedron, :legendre),
            (RefTriangle, :legendre),
            (RefTetrahedron, :dunavant),
            # (RefPrism, ...), # Not implemented yet (see discussion in #1007)
            # (RefPyramid, ...), # Not implement yet (see discussion in #1007)
        )
        for order in 1:3
            fqr = FacetQuadratureRule{refshape}(sym, order)
            fqr_default = FacetQuadratureRule{refshape}(order)
            for f in 1:nfacets(refshape)
                @test Ferrite.getweights(fqr, f) == Ferrite.getweights(fqr_default, f)
                @test Ferrite.getpoints(fqr, f) == Ferrite.getpoints(fqr_default, f)
            end
        end
    end

    @testset "polyorder" begin
        # Define a struct that returns a complete polynomial function with random weights.
        struct PolyFunction{dim}
            weights::Vector{Float64}
            exponents::Vector{NTuple{dim, Int}}
        end

        (f::PolyFunction{dim})(x::Vec{dim}) where {dim} = sum(w * sum(tuple(x...) .^ e) for (w, e) in zip(f.weights, f.exponents))

        function PolyFunction{1}(order::Int)
            return PolyFunction{1}(collect(range(0, 1, order + 1)), [(i,) for i in 0:order])
        end
        function PolyFunction{2}(order::Int)
            exponents = NTuple{2, Int}[]
            for i in 0:order
                for j in 0:(order - i)
                    push!(exponents, (i, j))
                end
            end
            return PolyFunction{2}(collect(range(0, 1, length(exponents))), exponents)
        end
        function PolyFunction{3}(order::Int)
            exponents = NTuple{3, Int}[]
            for i in 0:order
                for j in 0:(order - i)
                    for k in 0:(order - i - j)
                        push!(exponents, (i, j, k))
                    end
                end
            end
            return PolyFunction{3}(collect(range(0, 1, length(exponents))), exponents)
        end

        # Define functions to integrate over the difference reference shapes with hcubature.
        function integration_reference(::Type{Ferrite.RefHypercube{dim}}, f::F; kwargs...) where {dim, F}
            return hcubature(s -> f(Vec(s...)), -ones(Vec{dim}), ones(Vec{dim}); kwargs...)[1]
        end
        function integration_reference(::Type{RefTriangle}, f::F; kwargs...) where {F}
            duffy_transform(s) = Vec((s[1] * (1 - s[2]), s[2]))
            duffy_detJ(s) = 1 - s[2]
            return hcubature(s -> f(duffy_transform(s)) * duffy_detJ(s), zero(Vec{2}), ones(Vec{2}); kwargs...)[1]
        end
        function integration_reference(::Type{RefPrism}, f::F; kwargs...) where {F}
            duffy_transform(s) = Vec((s[1] * (1 - s[2]), s[2], s[3]))
            duffy_detJ(s) = 1 - s[2]
            return hcubature(s -> f(duffy_transform(s)) * duffy_detJ(s), zero(Vec{3}), ones(Vec{3}); kwargs...)[1]
        end
        function integration_reference(::Type{RefTetrahedron}, f::F; kwargs...) where {F}
            duffy_transform(s) = Vec((s[1], (1 - s[1]) * s[2], (1 - s[1]) * (1 - s[2]) * s[3]))
            duffy_detJ(s) = (1 - s[1])^2 * (1 - s[2])
            return hcubature(s -> f(duffy_transform(s)) * duffy_detJ(s), zero(Vec{3}), ones(Vec{3}); kwargs...)[1]
        end
        function integration_reference(::Type{RefPyramid}, f::F; kwargs...) where {F}
            duffy_transform(s) = Vec(s[1] * (1 - s[3]), s[2] * (1 - s[3]), s[3])
            duffy_detJ(s) = (1 - s[3])^2
            return hcubature(s -> f(duffy_transform(s)) * duffy_detJ(s), zero(Vec{3}), ones(Vec{3}); kwargs...)[1]
        end

        # Define function to integrate using the quadrature rule
        function integration_check(f::F, qr::QuadratureRule) where {F}
            s = zero(f(first(Ferrite.getpoints(qr))))
            for (ξ, w) in zip(Ferrite.getpoints(qr), Ferrite.getweights(qr))
                s += f(ξ) * w
            end
            return s
        end

        # Check that the polyorder kwarg gives the correct value
        for ((shape, type), polyorders) in [
                (RefLine, :legendre) => 1:10, (RefQuadrilateral, :legendre) => 1:3, (RefHexahedron, :legendre) => 1:3,
                (RefLine, :lobatto) => 1:10, (RefQuadrilateral, :lobatto) => 1:3, (RefHexahedron, :lobatto) => 1:3,
                (RefTriangle, :dunavant) => 1:8, (RefTriangle, :gaussjacobi) => 9:15,
                (RefTetrahedron, :jinyun) => 1:3, (RefTetrahedron, :keast_minimal) => 4:5, (RefTetrahedron, :keast_positive) => 4,
                (RefPrism, :polyquad) => 1:10, (RefPyramid, :polyquad) => 1:6,
            ]
            for polyorder in polyorders
                @testset "QuadratureRule{$shape}($type; polyorder = $polyorder)" begin
                    qr = QuadratureRule{shape}(type; polyorder)
                    f = PolyFunction{Ferrite.getrefdim(shape)}(polyorder)
                    solution = integration_reference(shape, f)
                    @test solution ≈ integration_check(f, qr)
                end
            end
        end
        qr_spec = QuadratureRule{RefQuadrilateral}(Float32; polyorder = 2)
        qr_def = QuadratureRule{RefQuadrilateral}(Float32, :legendre; polyorder = 2)
        @test eltype(Ferrite.getweights(qr_spec)) == Float32
        @test eltype(Ferrite.getpoints(qr_spec)) == Vec{2, Float32}
        @test Ferrite.getweights(qr_spec) == Ferrite.getweights(qr_def)
        @test Ferrite.getpoints(qr_spec) == Ferrite.getpoints(qr_def)

        qr_spec = QuadratureRule{RefQuadrilateral}(polyorder = 2)
        qr_def = QuadratureRule{RefQuadrilateral}(:legendre; polyorder = 2)
        @test eltype(Ferrite.getweights(qr_spec)) == Float64
        @test eltype(Ferrite.getpoints(qr_spec)) == Vec{2, Float64}
        @test Ferrite.getweights(qr_spec) == Ferrite.getweights(qr_def)
        @test Ferrite.getpoints(qr_spec) == Ferrite.getpoints(qr_def)
    end
end
