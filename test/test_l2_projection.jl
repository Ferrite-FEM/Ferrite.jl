
# Tests a L2-projection of integration point values (to nodal values),
# determined from the function y = 1 + x[1]^2 + (2x[2])^2
function test_projection(order, refshape)
    element = refshape == RefQuadrilateral ? Quadrilateral : Triangle
    grid = generate_grid(element, (1, 1), Vec((0.,0.)), Vec((1.,1.)))

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
    proj = L2Projector(ip, grid)
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
    @test_throws LinearAlgebra.PosDefException L2Projector(ip, grid; qr_lhs=QuadratureRule{refshape}(bad_order))
end

function make_mixedgrid_l2_tests()
    # generate a mesh with 2 quadrilateral and 2 triangular elements
    # 5 --- 6 --- 7 --- 8
    # |  1  | 2/3 |  4  |
    # 1 --- 2 --- 3 --- 4
    nodes = [Node(Float64.((x,y))) for (x, y) in
    #         1,      2,      3,      4,      5,      6,      7,      8
        ((0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1))]

    cells = [Quadrilateral((1, 2, 6, 5)), Triangle((2, 7, 6)), Triangle((2, 3, 7)), Quadrilateral((3, 4, 8, 7))]

    quadset = 1:1
    triaset = 2:3
    quadset_right = 4:4
    return Grid(cells, nodes), quadset, triaset, quadset_right
end

# Test a mixed grid, where only a subset of the cells contains a field
function test_projection_subset_of_mixedgrid()
    mesh, quadset, triaset, quadset_right = make_mixedgrid_l2_tests()

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
    qp_value = [f(spatial_coordinate(cv, qp, xe)) for qp in 1:getnquadpoints(cv)]
    qp_values = Vector{typeof(qp_value)}(undef, getncells(mesh))
    qp_values[1] = copy(qp_value)
    qp_values_matrix = fill(zero(eltype(qp_value)), getnquadpoints(cv), getncells(mesh))
    qp_values_matrix[:, 1] .= qp_value
    qp_values_dict = Dict(1 => copy(qp_value))

    # Now recover the nodal values using a L2 projection.
    # Assume f would only exist on the first cell, we project it to the nodes of the
    # 1st cell while ignoring the rest of the domain. NaNs should be stored in all
    # nodes that do not belong to the 1st cell
    proj = L2Projector(ip, mesh; set=quadset)
    point_vars = project(proj, qp_values, qr)
    point_vars_2 = project(proj, qp_values_matrix, qr)
    point_vars_3 = project(proj, qp_values_dict, qr)
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
    @test point_vars_3 ≈ ae

    # Do the same thing but for the triangle set
    ip = Lagrange{RefTriangle, order}()
    ip_geom = Lagrange{RefTriangle, 1}()
    qr = QuadratureRule{RefTriangle}(4)
    cv = CellValues(qr, ip, ip_geom)
    nqp = getnquadpoints(cv)

    qp_values_tria = [SymmetricTensor{2,2,Float64,3}[] for _ in 1:getncells(mesh)]
    qp_values_matrix_tria = [zero(SymmetricTensor{2,2}) * NaN for _ in 1:nqp, _ in 1:getncells(mesh)]
    qp_values_dict = Dict{Int, Vector{SymmetricTensor{2,2,Float64,3}}}()
    for (ic, cellid) in enumerate(triaset)
        xe = getcoordinates(mesh, cellid)
        # analytical values
        qp_values = [f(spatial_coordinate(cv, qp, xe)) for qp in 1:getnquadpoints(cv)]
        qp_values_tria[cellid] = qp_values
        qp_values_matrix_tria[:, cellid] .= qp_values
        qp_values_dict[cellid] = qp_values
    end

    #tria
    proj = L2Projector(ip, mesh; set=triaset)
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

function calculate_function_value_in_qpoints!(qp_data, sdh, cv, dofvector::Vector)
    for cell in CellIterator(sdh)
        qvector = qp_data[cellid(cell)]
        ae = dofvector[celldofs(cell)]
        resize!(qvector, getnquadpoints(cv))
        for q_point in 1:getnquadpoints(cv)
            qvector[q_point] = function_value(cv, q_point, ae)
        end
    end
    return qp_data
end

function test_add_projection_grid()
    grid = generate_grid(Triangle, (3,3))
    set1 = Set(1:getncells(grid)÷2)
    set2 = setdiff(1:getncells(grid), set1)

    dh = DofHandler(grid)
    ip = Lagrange{RefTriangle, 1}()
    sdh1 = SubDofHandler(dh, set1)
    add!(sdh1, :u, ip)
    sdh2 = SubDofHandler(dh, set2)
    add!(sdh2, :u, ip)
    close!(dh)

    solution = zeros(ndofs(dh))
    apply_analytical!(solution, dh, :u, x -> x[1]^2 - x[2]^2)

    qr = QuadratureRule{RefTriangle}(2)
    cv = CellValues(qr, ip, ip)

    # Fill qp_data with the interpolated values
    qp_data = [Float64[] for _ in 1:getncells(grid)]
    for (sdh, cv_) in ((sdh1, cv), (sdh2, cv))
        calculate_function_value_in_qpoints!(qp_data, sdh, cv_, solution)
    end

    # Build the first L2Projector with two different sets
    proj1 = L2Projector(grid)
    add!(proj1, set1, ip; qr_rhs = qr)
    add!(proj1, set2, ip; qr_rhs = qr)
    close!(proj1)

    # Build the second L2Projector with a single set using the convenience function
    proj2 = L2Projector(ip, grid)

    # Project both cases
    projected1 = project(proj1, qp_data)
    projected2 = project(proj2, qp_data, qr)

    # Evaluate at grid nodes to keep same numbering following the grid (dof distribution may be different)
    solution_at_nodes = evaluate_at_grid_nodes(dh, solution, :u)
    projected1_at_nodes = evaluate_at_grid_nodes(proj1, projected1)
    projected2_at_nodes = evaluate_at_grid_nodes(proj2, projected2)

    @test projected1_at_nodes ≈ solution_at_nodes
    @test projected2_at_nodes ≈ solution_at_nodes
end

function test_projection_mixedgrid()
    grid, quadset_left, triaset, quadset_right = make_mixedgrid_l2_tests()
    quadset_full = union(Set(quadset_left), quadset_right)
    @assert getncells(grid) == length(triaset) + length(quadset_full)
    # Test both for case with one cell excluded from projection, and will full grid included
    for quadset in (quadset_left, quadset_full)
        dh = DofHandler(grid)
        sdh_quad = SubDofHandler(dh, quadset)
        ip_quad = Lagrange{RefQuadrilateral, 1}()
        add!(sdh_quad, :u, ip_quad)
        sdh_tria = SubDofHandler(dh, triaset)
        ip_tria = Lagrange{RefTriangle, 1}()
        add!(sdh_tria, :u, ip_tria)
        close!(dh)

        solution = zeros(ndofs(dh))
        apply_analytical!(solution, dh, :u, x -> x[1]^2 - x[2]^2)

        qr_quad = QuadratureRule{RefQuadrilateral}(2)
        cv_quad = CellValues(qr_quad, ip_quad, ip_quad)
        qr_tria = QuadratureRule{RefTriangle}(2)
        cv_tria = CellValues(qr_tria, ip_tria, ip_tria)

        # Fill qp_data with the interpolated values
        qp_data = [Float64[] for _ in 1:getncells(grid)]
        for (sdh, cv) in ((sdh_quad, cv_quad), (sdh_tria, cv_tria))
            calculate_function_value_in_qpoints!(qp_data, sdh, cv, solution)
        end

        # Finally, let's build the L2Projector and check if we can project back the solution
        proj = L2Projector(grid)
        add!(proj, triaset, ip_tria; qr_rhs = qr_tria)
        add!(proj, quadset, ip_quad; qr_rhs = qr_quad)
        close!(proj)

        # Quadrature rules must be in the same order as ip's are added to proj.
        projected = project(proj, qp_data)

        # Evaluate at grid nodes to keep same numbering following the grid (dof distribution may be different)
        solution_at_nodes = evaluate_at_grid_nodes(dh, solution, :u)
        projected_at_nodes = evaluate_at_grid_nodes(proj, projected)

        # Since one part of the grid is excluded, nodes in this region will be NaN.
        # So we only want to check those nodes attached to cells in the cellsets.
        active_nodes = Set{Int}()
        for cell in CellIterator(grid, union(quadset, triaset))
            for n in Ferrite.getnodes(cell)
                push!(active_nodes, n)
            end
        end
        check_nodes = collect(active_nodes)

        @test projected_at_nodes[check_nodes] ≈ solution_at_nodes[check_nodes]

    end
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
        fname = joinpath(tmp, "projected")
        VTKGridFile(fname, grid) do vtk
            write_projection(vtk, p, p_scalar, "p_scalar")
            write_projection(vtk, p, p_vec, "p_vec")
            write_projection(vtk, p, p_tens, "p_tens")
            write_projection(vtk, p, p_stens, "p_stens")
        end
        # The following test may fail due to floating point inaccuracies
        # These could occur due to e.g. changes in system architecture.
        if Sys.islinux() && Sys.ARCH === :x86_64
            @test bytes2hex(open(SHA.sha1, fname*".vtu", "r")) == (
                subset ? "b3fef3de9f38ca9ddd92f2f67a1606d07ca56d67" :
                         "bc2ec8f648f9b8bccccf172c1fc48bf03340329b"
            )
        end
    end

end

function test_show_l2()
    grid = generate_grid(Triangle, (2,2))
    ip = Lagrange{RefTriangle, 1}()
    proj = L2Projector(ip, grid)
    @test repr("text/plain", proj) == repr(typeof(proj)) * "\n  projection on:           8/8 cells in grid\n  function interpolation:  Lagrange{RefTriangle, 1}()\n  geometric interpolation: Lagrange{RefTriangle, 1}()\n"

    # Multi-domain setup
    proj2 = L2Projector(grid)
    @test sprint(show, MIME"text/plain"(), proj2) == "L2Projector (not closed)"
    qr_rhs = QuadratureRule{RefTriangle}(2)
    add!(proj2, Set(1:2), ip; qr_rhs)
    add!(proj2, Set(3:4), ip; qr_rhs)
    close!(proj2)
    showstr = sprint(show, MIME"text/plain"(), proj2)
    @test contains(showstr, "L2Projector")
    @test contains(showstr, "4/8 cells in grid")
    @test contains(showstr, "Split into 2 sets")
end

function test_l2proj_errorpaths()
    grid = generate_grid(Triangle, (2,3))
    ip = Lagrange{RefTriangle, 1}()
    proj = L2Projector(grid)                        # Multiple subdomains
    proj1 = L2Projector(ip, grid; set=collect(1:4)) # Single sub-domain case
    qr_tria = QuadratureRule{RefTriangle}(2)
    qr_quad = QuadratureRule{RefQuadrilateral}(2)

    # Providing wrong quadrature rules
    exception_rhs = ErrorException("The reference shape of the interpolation and the qr_rhs must be the same")
    exception_lhs = ErrorException("The reference shape of the interpolation and the qr_lhs must be the same")
    @test_throws exception_rhs add!(proj, Set(1:2), ip; qr_rhs = qr_quad)
    @test_throws exception_lhs add!(proj, Set(1:2), ip; qr_rhs = qr_tria, qr_lhs = qr_quad)

    # Build up a 2-domain case
    add!(proj, Set(1:2), ip; qr_rhs = qr_tria)
    add!(proj, Set(3:4), ip; qr_rhs = qr_tria)
    data_valid = Dict(i => rand(getnquadpoints(qr_tria)) for i in 1:4)

    # Try projecting when not closed
    @test_throws ErrorException("The L2Projector is not closed") project(proj, data_valid)
    close!(proj)

    # Not giving quadrature rule
    noquad_exception = ErrorException("The right-hand-side quadrature rule must be provided, unless already given to the L2Projector")
    @test_throws noquad_exception project(proj1, data_valid)
    # Providing wrong quadrature rule to project
    wrongquad_exception = ErrorException("Reference shape of quadrature rule and cells doesn't match. Please ensure that `qrs_rhs` has the same order as sets are added to the L2Projector")
    @test_throws wrongquad_exception project(proj1, data_valid, qr_quad)

    # Giving data indexed by set index instead of cell index
    data_invalid = [rand(getnquadpoints(qr_tria)) for _ in 1:4]
    invalid_data_exception = ErrorException("vars is indexed by the cellid, not the index in the set: length(vars) != number of cells")
    @test_throws invalid_data_exception project(proj1, data_invalid, qr_tria)
    # Giving data with too many or too few quadrature points
    data_invalid2 = [rand(getnquadpoints(qr_tria) + 1) for _ in 1:getncells(grid)]
    data_invalid3 = [rand(getnquadpoints(qr_tria) - 1) for _ in 1:getncells(grid)]
    wrongnqp_exception = ErrorException("The number of variables per cell doesn't match the number of quadrature points")
    @test_throws wrongnqp_exception project(proj1, data_invalid2, qr_tria)
    @test_throws wrongnqp_exception project(proj1, data_invalid3, qr_tria)

end

@testset "Test L2-Projection" begin
    test_projection(1, RefQuadrilateral)
    test_projection(1, RefTriangle)
    test_projection(2, RefQuadrilateral)
    test_projection(2, RefTriangle)
    test_projection_subset_of_mixedgrid()
    test_add_projection_grid()
    test_projection_mixedgrid()
    test_export(subset=false)
    test_export(subset=true)
    test_show_l2()
    test_l2proj_errorpaths()
end
