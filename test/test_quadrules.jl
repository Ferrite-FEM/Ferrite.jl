@testset "Quadrature testing" begin
    ref_tet_vol(dim) = 1 / factorial(dim)
    ref_square_vol(dim) = 2^dim

    function integrate(qr::QuadratureRule, f::Function)
        I = 0.0
        for (w, x) in zip(Ferrite.getweights(qr), Ferrite.getpoints(qr))
            I += w * f(x)
        end
        return I
    end

    # Hypercube
    for (dim, shape) = ((1, RefLine), (2, RefQuadrilateral), (3, RefHexahedron))
        for order in (1,2,3,4)
            f = (x, p) -> sum([x[i]^p for i in 1:length(x)])
            # Legendre
            qr = QuadratureRule{shape}(:legendre, order)
            @test integrate(qr, (x) -> f(x, 2*order-1)) < 1e-14
            @test sum(qr.weights) ≈ ref_square_vol(dim)
            @test sum(Ferrite.getweights(qr)) ≈ ref_square_vol(dim)
            # Lobatto
            if order > 1
                qr = QuadratureRule{shape}(:lobatto, order)
                @test integrate(qr, (x) -> f(x, 2*order-1)) < 1e-14
                @test sum(qr.weights) ≈ ref_square_vol(dim)
                @test sum(Ferrite.getweights(qr)) ≈ ref_square_vol(dim)
            end
        end
    end
    @test_throws ArgumentError QuadratureRule{RefLine}(:einstein, 2)

    # Tetrahedron
    g = (x) -> sqrt(sum(x))
    dim = 2
    for order in 1:15
        qr = QuadratureRule{RefTriangle}(:legendre, order)
        # http://www.wolframalpha.com/input/?i=integrate+sqrt(x%2By)+from+x+%3D+0+to+1,+y+%3D+0+to+1-x
        @test integrate(qr, g) - 0.4 < 0.01
        @test sum(qr.weights) ≈ ref_tet_vol(dim)
    end
    @test_throws ArgumentError QuadratureRule{RefTriangle}(:einstein, 2)
    @test_throws ArgumentError QuadratureRule{RefTriangle}(0)

    dim = 3
    for order in (1, 2, 3, 4)
        qr = QuadratureRule{RefTetrahedron}(:legendre, order)
        # Table 1:
        # http://www.m-hikari.com/ijma/ijma-2011/ijma-1-4-2011/venkateshIJMA1-4-2011.pdf
        @test integrate(qr, g) - 0.14 < 0.01
        @test sum(qr.weights) ≈ ref_tet_vol(dim)
    end
    @test_throws ArgumentError QuadratureRule{RefTetrahedron}(:einstein, 2)
    @test_throws ArgumentError QuadratureRule{RefTetrahedron}(0)

    @testset "Quadrature rules for $ref_cell" for ref_cell in (
        Line,
        Quadrilateral,
        Triangle,
        Hexahedron,
        Tetrahedron,
        Wedge,
        Pyramid)

        refshape = ref_cell.super.parameters[1]
        dim = refshape.super.parameters[1]

        dim > 1 && @testset "$refshape face-cell spatial coordinates" begin
            grid = generate_grid(ref_cell, ntuple(_->3, dim))
            for cellid in 1:getncells(grid)
                cell = grid.cells[cellid]
                ccoords = getcoordinates(grid, cellid)
                Vec_t = Vec{dim,Float64}
                Vec_face_t = Vec{dim-1,Float64}
                for lfaceid in Ferrite.nfacets(refshape)
                    facenodes = Ferrite.facets(cell)[lfaceid]
                    fcoords = zeros(Vec_t, length(facenodes))
                    for (i,nodeid) in enumerate(facenodes)
                        x = grid.nodes[nodeid].x
                        fcoords[i] = x
                    end
                    ipcell = Lagrange{refshape,1}()
                    ipface = Lagrange{getfacerefshape(cell,lfaceid),1}()

                    ξface = rand(Vec_face_t)/4
                    ξcell = Ferrite.face_to_element_transformation(ξface, refshape, lfaceid)

                    xface = zero(Vec_t)
                    for i in 1:getnbasefunctions(ipface)
                        xface += Ferrite.shape_value(ipface, ξface, i) * fcoords[i]
                    end

                    xcell = zero(Vec_t)
                    for i in 1:getnbasefunctions(ipcell)
                        xcell += shape_value(ipcell, ξcell, i) * ccoords[i]
                    end

                    @test xcell ≈ xface
                end
            end
        end

        @testset "$ref_cell unknown face error path" begin
            for face in (-1, 0, 100)
                err = ArgumentError("unknown face number")
                @test_throws err Ferrite.weighted_normal(Tensor{2,dim}(zeros(dim^2)), refshape, face)
                pt = Vec{dim-1, Float64}(ntuple(i -> 0.0, dim-1))
                @test_throws err Ferrite.face_to_element_transformation(pt, refshape, face)
            end
        end
    end
end
