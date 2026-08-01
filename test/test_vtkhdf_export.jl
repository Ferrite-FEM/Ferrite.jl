using Ferrite, Test
using VTKHDF
using VTKHDF.HDF5

@testset "VTKHDFGridFile" begin
    grid = generate_grid(Quadrilateral, (4, 3))
    ip = Lagrange{RefQuadrilateral, 1}()
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    add!(dh, :v, ip^2)
    close!(dh)
    u = rand(ndofs(dh))
    dir = mktempdir()

    @testset "static" begin
        fn = joinpath(dir, "static.vtkhdf")
        addcellset!(grid, "left", x -> x[1] < 0)
        v = VTKHDFGridFile(fn, dh) do vtk
            write_solution(vtk, dh, u)
            write_cell_data(vtk, collect(1:getncells(grid)), "cellid")
            write_node_data(vtk, rand(getnnodes(grid)), "nd")
            Ferrite.write_cellset(vtk, grid, "left")
        end
        @test v isa VTKHDFGridFile
        h5open(fn) do f
            g = f["VTKHDF"]
            @test attrs(g)["Type"] == "UnstructuredGrid"
            @test size(g["Points"]) == (3, getnnodes(grid))
            @test length(g["PointData/u"]) == getnnodes(grid)
            @test size(g["PointData/v"]) == (3, getnnodes(grid))  # 2D Vec padded to 3
            @test read(g["CellData/cellid"]) == 1:getncells(grid)
            @test haskey(g, "CellData/left")
        end
    end

    @testset "temporal, grid stored once" begin
        fn = joinpath(dir, "temporal.vtkhdf")
        vtkhdf = VTKHDFGridFile(fn, dh; temporal = true)
        nsteps = 4
        for (step, t) in enumerate(range(0.0, 1.5; length = nsteps))
            write_timestep(vtkhdf, t) do vtk
                write_solution(vtk, dh, u .* step)
            end
        end
        close(vtkhdf)
        h5open(fn) do f
            g = f["VTKHDF"]
            @test size(g["Points"]) == (3, getnnodes(grid))       # exactly one copy
            @test attrs(g["Steps"])["NSteps"] == nsteps
            @test length(g["PointData/u"]) == nsteps * getnnodes(grid)
            @test read(g["Steps/PointOffsets"]) == zeros(nsteps)
        end
    end

    @testset "discontinuous" begin
        dhdg = DofHandler(grid)
        add!(dhdg, :w, DiscontinuousLagrange{RefQuadrilateral, 1}())
        close!(dhdg)
        fn = joinpath(dir, "dg.vtkhdf")
        VTKHDFGridFile(fn, dhdg) do vtk
            write_solution(vtk, dhdg, rand(ndofs(dhdg)))
        end
        h5open(fn) do f
            # nodes are duplicated per cell
            @test size(f["VTKHDF/Points"]) == (3, sum(Ferrite.nnodes, getcells(grid)))
        end
    end

    @testset "projection" begin
        qr = QuadratureRule{RefQuadrilateral}(2)
        proj = L2Projector(ip, grid)
        qp_data = [[norm(x) for x in Ferrite.getpoints(qr)] for _ in 1:getncells(grid)]
        projected = project(proj, qp_data, qr)
        fn = joinpath(dir, "proj.vtkhdf")
        VTKHDFGridFile(fn, grid) do vtk
            write_projection(vtk, proj, projected, "p")
        end
        h5open(fn) do f
            @test length(f["VTKHDF/PointData/p"]) == getnnodes(grid)
        end
    end
end
