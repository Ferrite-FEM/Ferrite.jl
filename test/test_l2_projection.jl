
# Tests a L2-projection of integration point values (to nodal values),
# determined from the function y = 1 + x[1]^2 + (2x[2])^2
function test_projection(order, refshape)
    element = refshape == RefQuadrilateral ? Quadrilateral : Triangle
    grid = generate_grid(element, (1, 1), Vec((0.,0.)), Vec((1.,1.)))
    Ferrite.transform_coordinates!(grid, x->x+rand(Vec{2,Float64})*0.01)

    ip = Lagrange{refshape, order}()
    ip_geom = Lagrange{refshape, 1}()
    qr = Ferrite._mass_qr(ip)
    cv = CellValues(qr, ip, ip_geom)

    # Create node values for the cell
    f(x) = 1 + x[1]^2 + (2x[2])^2
    # Nodal approximations for this simple grid when using linear interpolation
    f_approx(i) = refshape == RefQuadrilateral ?
        [0.1666666666666664, 1.166666666666667, 5.166666666666667, 4.166666666666666][i] :
        [0.444444444444465, 1.0277777777778005, 4.027777777777753, 5.444444444444435][i]

    # analytical values
    function analytical(f)
        qp_values = []
        for cell in CellIterator(grid)
            reinit!(cv, cell)
            r = [f(spatial_coordinate(cv, qp, getcoordinates(cell))) for qp in 1:getnquadpoints(cv)]
            push!(qp_values, r)
        end
        return identity.(qp_values) # Tighten the type
    end

    qp_values = analytical(f)

    # Now recover the nodal values using a L2 projection.
    proj = L2Projector(ip, grid; geom_ip=ip_geom)
    point_vars = project(proj, qp_values, qr)
    qp_values_matrix = reduce(hcat, qp_values)
    point_vars_2 = project(proj, qp_values_matrix, qr)
    if order == 1
        # A linear approximation can not recover a quadratic solution,
        # so projected values will be different from the analytical ones
        ae = [f_approx(i) for i in 1:4]
    elseif order == 2
        # For a quadratic approximation the analytical solution is recovered
        ae = zeros(length(point_vars))
        apply_analytical!(ae, proj.dh, :_, f)
    end
    @test point_vars ≈ point_vars_2 ≈ ae

    # Vec
    f_vector(x) = Vec{1,Float64}((f(x),))
    qp_values = analytical(f_vector)
    point_vars = project(proj, qp_values, qr)
    if order == 1
        ae = [Vec{1,Float64}((f_approx(j),)) for j in 1:4]
    elseif order == 2
        ae = zeros(length(point_vars))
        apply_analytical!(ae, proj.dh, :_, x -> f_vector(x)[1])
        ae = reinterpret(Vec{1,Float64}, ae)
    end
    @test point_vars ≈ ae

    # Tensor
    f_tensor(x) = Tensor{2,2,Float64}((f(x),2*f(x),3*f(x),4*f(x)))
    qp_values = analytical(f_tensor)
    qp_values_matrix = reduce(hcat, qp_values)::Matrix
    point_vars = project(proj, qp_values, qr)
    point_vars_2 = project(proj, qp_values_matrix, qr)
    if order == 1
        ae = [Tensor{2,2,Float64}((f_approx(i),2*f_approx(i),3*f_approx(i),4*f_approx(i))) for i in 1:4]
    elseif order == 2
        ae = zeros(4, length(point_vars))
        for i in 1:4
            apply_analytical!(@view(ae[i, :]), proj.dh, :_, x -> f_tensor(x)[i])
        end
        ae = reinterpret(reshape, Tensor{2,2,Float64,4}, ae)
    end
    @test point_vars ≈ point_vars_2 ≈ ae

    # SymmetricTensor
    f_stensor(x) = SymmetricTensor{2,2,Float64}((f(x),2*f(x),3*f(x)))
    qp_values = analytical(f_stensor)
    qp_values_matrix = reduce(hcat, qp_values)
    point_vars = project(proj, qp_values, qr)
    point_vars_2 = project(proj, qp_values_matrix, qr)
    if order == 1
        ae = [SymmetricTensor{2,2,Float64}((f_approx(i),2*f_approx(i),3*f_approx(i))) for i in 1:4]
    elseif order == 2
        ae = zeros(3, length(point_vars))
        for i in 1:3
            apply_analytical!(@view(ae[i, :]), proj.dh, :_, x -> f_stensor(x).data[i])
        end
        ae = reinterpret(reshape, SymmetricTensor{2,2,Float64,3}, ae)
    end
    @test point_vars ≈ point_vars_2 ≈ ae

    # Test error-path with bad qr
    if refshape == RefTriangle && order == 2
        bad_order = 2
    else
        bad_order = 1
    end
    @test_throws LinearAlgebra.PosDefException L2Projector(ip, grid; qr_lhs=QuadratureRule{refshape}(bad_order), geom_ip=ip_geom)
end

# Test a mixed grid, where only a subset of the cells contains a field
function test_projection_mixedgrid()
    # generate a mesh with 1 quadrilateral and 2 triangular elements
    dim = 2
    nodes = Node{dim, Float64}[]
    push!(nodes, Node((0.0, 0.0)))
    push!(nodes, Node((1.0, 0.0)))
    push!(nodes, Node((2.0, 0.0)))
    push!(nodes, Node((0.0, 1.0)))
    push!(nodes, Node((1.0, 1.0)))
    push!(nodes, Node((2.0, 1.0)))

    cells = Ferrite.AbstractCell[]
    push!(cells, Quadrilateral((1,2,5,4)))
    push!(cells, Triangle((2,3,6)))
    push!(cells, Triangle((2,6,5)))

    quadset = 1:1
    triaset = 2:3
    mesh = Grid(cells, nodes)

    order = 2
    ip = Lagrange{RefQuadrilateral, order}()
    ip_geom = Lagrange{RefQuadrilateral, 1}()
    qr = QuadratureRule{RefQuadrilateral}(order+1)
    cv = CellValues(qr, ip, ip_geom)

    # Create node values for the 1st cell
    # use a SymmetricTensor here for testing the symmetric version of project
    f(x) = SymmetricTensor{2,2,Float64}((1 + x[1]^2, 2x[2]^2, x[1]*x[2]))
    xe = getcoordinates(mesh, 1)
    
    # analytical values
    qp_values = [[f(spatial_coordinate(cv, qp, xe)) for qp in 1:getnquadpoints(cv)]]
    qp_values_matrix = reduce(hcat, qp_values)

    # Now recover the nodal values using a L2 projection.
    # Assume f would only exist on the first cell, we project it to the nodes of the
    # 1st cell while ignoring the rest of the domain. NaNs should be stored in all
    # nodes that do not belong to the 1st cell
    proj = L2Projector(ip, mesh; geom_ip=ip_geom, set=quadset)
    point_vars = project(proj, qp_values, qr)
    point_vars_2 = project(proj, qp_values_matrix, qr)
    projection_at_nodes = evaluate_at_grid_nodes(proj, point_vars)
    for cellid in quadset
        for nodeid in mesh.cells[cellid].nodes
            x = mesh.nodes[nodeid].x
            @test projection_at_nodes[nodeid] ≈ f(x)
        end
    end

    ae = zeros(3, length(point_vars))
    for i in 1:3
        apply_analytical!(@view(ae[i, :]), proj.dh, :_, x -> f(x).data[i], quadset)
    end
    ae = reinterpret(reshape, SymmetricTensor{2,2,Float64,3}, ae)
    @test point_vars ≈ point_vars_2 ≈ ae

    # Do the same thing but for the triangle set
    ip = Lagrange{RefTriangle, order}()
    ip_geom = Lagrange{RefTriangle, 1}()
    qr = QuadratureRule{RefTriangle}(4)
    cv = CellValues(qr, ip, ip_geom)
    nqp = getnquadpoints(cv)

    qp_values_tria = [zeros(SymmetricTensor{2,2}, nqp) for _ in triaset]
    qp_values_matrix_tria = [zero(SymmetricTensor{2,2}) for _ in 1:nqp, _ in triaset]
    for (ic, cellid) in enumerate(triaset)
        xe = getcoordinates(mesh, cellid)
        # analytical values
        qp_values = [f(spatial_coordinate(cv, qp, xe)) for qp in 1:getnquadpoints(cv)]
        qp_values_tria[ic] = qp_values
        qp_values_matrix_tria[:, ic] .= qp_values
    end

    #tria
    proj = L2Projector(ip, mesh; geom_ip=ip_geom, set=triaset)
    point_vars = project(proj, qp_values_tria, qr)
    point_vars_2 = project(proj, qp_values_matrix_tria, qr)
    projection_at_nodes = evaluate_at_grid_nodes(proj, point_vars)
    for cellid in triaset
        for nodeid in mesh.cells[cellid].nodes
            x = mesh.nodes[nodeid].x
            @test projection_at_nodes[nodeid] ≈ f(x)
        end
    end

    ae = zeros(3, length(point_vars))
    for i in 1:3
        apply_analytical!(@view(ae[i, :]), proj.dh, :_, x -> f(x).data[i], triaset)
    end
    ae = reinterpret(reshape, SymmetricTensor{2,2,Float64,3}, ae)
    @test point_vars ≈ point_vars_2 ≈ ae
end

function test_export(;subset::Bool)
    grid = generate_grid(Quadrilateral, (2, 1))
    qr = QuadratureRule{RefQuadrilateral}(2)
    ip = Lagrange{RefQuadrilateral,1}()
    cv = CellValues(qr, ip)
    nqp = getnquadpoints(cv)
    qpdata_scalar = [zeros(nqp) for _ in 1:getncells(grid)]
    qpdata_vec = [zeros(Vec{2}, nqp) for _ in 1:getncells(grid)]
    qpdata_tens = [zeros(Tensor{2,2}, nqp) for _ in 1:getncells(grid)]
    qpdata_stens = [zeros(SymmetricTensor{2,2}, nqp) for _ in 1:getncells(grid)]
    function f(x)
        if subset && x[1] > 0.001
            return NaN
        else
            return 2x[1] + x[2]
        end
    end
    for cell in CellIterator(grid)
        reinit!(cv, cell)
        xh = getcoordinates(cell)
        for qp in 1:getnquadpoints(cv)
            x = spatial_coordinate(cv, qp, xh)
            qpdata_scalar[cellid(cell)][qp] = f(x)
            qpdata_vec[cellid(cell)][qp] = Vec{2}(i -> i * f(x))
            qpdata_tens[cellid(cell)][qp] = Tensor{2,2}((i,j) -> i * j * f(x))
            qpdata_stens[cellid(cell)][qp] = SymmetricTensor{2,2}((i,j) -> i * j * f(x))
        end
    end
    p = subset ? L2Projector(ip, grid; set=1:1) : L2Projector(ip, grid)
    p_scalar = project(p, qpdata_scalar, qr)::Vector{Float64}
    p_vec = project(p, qpdata_vec, qr)::Vector{<:Vec{2}}
    p_tens = project(p, qpdata_tens, qr)::Vector{<:Tensor{2,2}}
    p_stens = project(p, qpdata_stens, qr)::Vector{<:SymmetricTensor{2,2}}

    # reshaping for export with evaluate_at_grid_nodes
    fnodes = [f(x.x) for x in grid.nodes]
    nindex = isnan.(fnodes)
    findex = (!isnan).(fnodes)
    let r = evaluate_at_grid_nodes(p, p_scalar),
        rv = Ferrite._evaluate_at_grid_nodes(p, p_scalar, Val(true))
        @test size(r) == (6,)
        @test all(isnan, r[nindex])
        @test all(isnan, rv[nindex])
        @test r[findex] ≈ fnodes[findex]
        @test rv isa Matrix{Float64}
        @test r isa Vector{Float64}
        @test r[findex] == vec(rv)[findex]
    end
    let r = evaluate_at_grid_nodes(p, p_vec),
        rv = Ferrite._evaluate_at_grid_nodes(p, p_vec, Val(true))
        @test size(r) == (6,)
        @test getindex.(r[findex], 1) ≈  fnodes[findex]
        @test getindex.(r[findex], 2) ≈ 2fnodes[findex]
        @test all(y -> all(isnan, y), r[nindex])
        @test rv[1:2, findex] ≈ reshape(reinterpret(Float64, r), (2, 6))[:, findex]
        @test all(iszero, rv[3:3, findex])
        @test all(isnan, rv[:, nindex])
    end
    let r = evaluate_at_grid_nodes(p, p_tens),
        rv = Ferrite._evaluate_at_grid_nodes(p, p_tens, Val(true))
        @test size(r) == (6,)
        @test getindex.(r[findex], 1) ≈  fnodes[findex] # 11-components
        @test getindex.(r[findex], 2) ≈ 2fnodes[findex] # 12-components
        @test getindex.(r[findex], 3) ≈ 2fnodes[findex] # 21-components
        @test getindex.(r[findex], 4) ≈ 4fnodes[findex] # 22-components
        @test all(y -> all(isnan, y), r[nindex])
        voigt_perm = [1, 4, 3, 2]
        @test rv[voigt_perm, findex] ≈ reshape(reinterpret(Float64, r), (4, 6))[:, findex]
        @test all(isnan, rv[:, nindex])
    end
    let r = evaluate_at_grid_nodes(p, p_stens),
        rv = Ferrite._evaluate_at_grid_nodes(p, p_stens, Val(true))
        @test size(r) == (6,)
        @test getindex.(r[findex], 1) ≈  fnodes[findex] # 11-components
        @test getindex.(r[findex], 2) ≈ 2fnodes[findex] # 21-components
        @test getindex.(r[findex], 3) ≈ 2fnodes[findex] # 12-components
        @test getindex.(r[findex], 4) ≈ 4fnodes[findex] # 22-components
        @test all(y -> all(isnan, y), r[nindex])
        voigt_perm = [1, 3, 2]
        @test rv[voigt_perm, findex] ≈ reshape(reinterpret(Float64, r), (3, 6))[:, findex]
        @test all(isnan, rv[:, nindex])
    end

    mktempdir() do tmp
        fname = vtk_grid(joinpath(tmp, "projected"), grid) do vtk
            vtk_point_data(vtk, p, p_scalar, "p_scalar")
            vtk_point_data(vtk, p, p_vec, "p_vec")
            vtk_point_data(vtk, p, p_tens, "p_tens")
            vtk_point_data(vtk, p, p_stens, "p_stens")
        end
        @test bytes2hex(open(SHA.sha1, fname[1], "r")) in (
            subset ? ("261cfe21de7a478e14f455e783694651a91eeb60", "b3fef3de9f38ca9ddd92f2f67a1606d07ca56d67") :
                     ("3b8ffb444db1b4cee1246a751da88136116fe49b", "bc2ec8f648f9b8bccccf172c1fc48bf03340329b")
        )
    end
end

function test_show()
    grid = generate_grid(Triangle, (2,2))
    ip = Lagrange{RefTriangle, 1}()
    proj = L2Projector(ip, grid)
    @test repr("text/plain", proj) == repr(typeof(proj)) * "\n  projection on:           8/8 cells in grid\n  function interpolation:  Lagrange{RefTriangle, 1}()\n  geometric interpolation: Lagrange{RefTriangle, 1}()\n"
end

@testset "Test L2-Projection" begin
    test_projection(1, RefQuadrilateral)
    test_projection(1, RefTriangle)
    test_projection(2, RefQuadrilateral)
    test_projection(2, RefTriangle)
    test_projection_mixedgrid()
    test_export(subset=false)
    test_export(subset=true)
    test_show()
end
