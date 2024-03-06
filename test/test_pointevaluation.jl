function scalar_field()
    # isoparametric approximation
    mesh = generate_grid(QuadraticQuadrilateral, (20, 20))
    f(x) = x[1]^2

    ip_f = Lagrange{RefQuadrilateral,2}() # function interpolation
    ip_g = Lagrange{RefQuadrilateral,2}() # geometry interpolation

    # compute values in quadrature points
    qr = QuadratureRule{RefQuadrilateral}(3) # exactly approximate quadratic field
    cv = CellValues(qr, ip_f, ip_g)
    qp_vals = [Vector{Float64}(undef, getnquadpoints(cv)) for _ in 1:getncells(mesh)]
    for cellid in eachindex(mesh.cells)
        xe = getcoordinates(mesh, cellid)
        reinit!(cv, xe)
        for qp in 1:getnquadpoints(cv)
            qp_vals[cellid][qp] = f(spatial_coordinate(cv, qp, xe))
        end
    end

    # do a L2Projection for getting values in dofs
    projector = L2Projector(ip_f, mesh)
    projector_vals = project(projector, qp_vals, qr)

    # points where we want to retrieve field values
    points = [Vec((x, 0.52)) for x in range(0.0; stop=1.0, length=100)]

    # set up PointEvalHandler and retrieve values
    ph = PointEvalHandler(mesh, points)
    vals = evaluate_at_points(ph, projector, projector_vals)
    @test f.(points) ≈ vals

    # alternatively retrieve vals from nodal values TODO: make this work?
    # vals = evaluate_at_points(ph, nodal_vals)
    # @test f.(points) ≈ vals
end

function vector_field()
    ## vector field
    # isoparametric approximation
    mesh = generate_grid(QuadraticQuadrilateral, (20, 20))
    f(x) = Vec((x[1]^2, x[1]))
    nodal_vals = [f(p.x) for p in mesh.nodes]

    ip_f = Lagrange{RefQuadrilateral,2}() # function interpolation
    ip_g = Lagrange{RefQuadrilateral,2}() # geometry interpolation

    # compute values in quadrature points
    qr = QuadratureRule{RefQuadrilateral}(3) # exactly approximate quadratic field
    cv = CellValues(qr, ip_f, ip_g)
    qp_vals = [Vector{Vec{2,Float64}}(undef, getnquadpoints(cv)) for i=1:getncells(mesh)]
    for cellid in eachindex(mesh.cells)
        xe = getcoordinates(mesh, cellid)
        reinit!(cv, xe)
        for qp in 1:getnquadpoints(cv)
            qp_vals[cellid][qp] = f(spatial_coordinate(cv, qp, xe))
        end
    end

    # do a L2Projection for getting values in dofs
    projector = L2Projector(ip_f, mesh)
    projector_vals = project(projector, qp_vals, qr)
    # projector_vals = convert(Vector{Float64}, reinterpret(Float64, projector_vals))

    # points where we want to retrieve field values
    points = [Vec((x, 0.52)) for x in range(0.0; stop=1.0, length=100)]

    # set up PointEvalHandler and retrieve values
    ph = PointEvalHandler(mesh, points)
    vals = evaluate_at_points(ph, projector, projector_vals)
    @test f.(points) ≈ vals

    # alternatively retrieve vals from nodal values# TODO
    # vals = evaluate_at_points(ph, nodal_vals)
    # @test f.(points) ≈ vals
end

function superparametric()
    # superparametric approximation
    mesh = generate_grid(Quadrilateral, (20, 20))
    f(x) = x*x[1]
    ip_f = Lagrange{RefQuadrilateral,2}() # function interpolation

    # compute values in quadrature points
    qr = QuadratureRule{RefQuadrilateral}(3) # exactly approximate quadratic field
    cv = CellValues(qr, ip_f)
    qp_vals = [Vector{Vec{2,Float64}}(undef, getnquadpoints(cv)) for i=1:getncells(mesh)]
    for cellid in eachindex(mesh.cells)
        xe = getcoordinates(mesh, cellid)
        reinit!(cv, xe)
        for qp in 1:getnquadpoints(cv)
            qp_vals[cellid][qp] = f(spatial_coordinate(cv, qp, xe))
        end
    end

    # do a L2Projection for getting values in dofs
    projector = L2Projector(ip_f, mesh)
    projector_vals = project(projector, qp_vals, qr)

    # points where we want to retrieve field values
    points = [Vec((x, 0.52)) for x in range(0.0; stop=1.0, length=100)]

    # set up PointEvalHandler and retrieve values
    ph = PointEvalHandler(mesh, points)
    vals = evaluate_at_points(ph, projector, projector_vals)

    # can recover a quadratic field by a quadratic approximation
    @test f.(points) ≈ vals
end

function dofhandler()
    mesh = generate_grid(Quadrilateral, (2,2))
    dof_vals = [1., 2., 5., 4., 3., 6., 8., 7., 9.]
    points = [node.x for node in mesh.nodes] # same as nodes

    dh = DofHandler(mesh)
    add!(dh, :s, Lagrange{RefQuadrilateral,1}()) # a scalar field
    close!(dh)

    ph = PointEvalHandler(mesh, points)
    vals = evaluate_at_points(ph, dh, dof_vals, :s)
    @test vals ≈ 1.0:9.0

    # TODO
    # vals = evaluate_at_points(ph, collect(1.0:9.0))
    # @test vals ≈ 1.0:9.0
end

function dofhandler2(;three_dimensional=true)
    # Computes the L2 projection of a quadratic field exactly
    # but not using L2Projector since we want the DofHandler dofs
    if (three_dimensional)
        mesh = generate_grid(Hexahedron, (10, 10, 10))
        f_s = x -> 1.0 + x[1] + x[2] + x[1] * x[2] + x[2] * x[3]
        f_v = x ->  Vec{3}((1.0 + x[1] + x[2] + x[1] * x[2], 2.0 - x[1] - x[2] - x[1] * x[2], 4.0 + x[1] - x[2] + x[3] - x[1] * x[3] - x[2] * x[3]))
        points = [Vec((x, x, x)) for x in range(0; stop=1, length=100)]
        ip_f = Lagrange{RefHexahedron,2}()
        ip_f_v = ip_f^3
        qr = QuadratureRule{RefHexahedron}(3)
    else 
        mesh = generate_grid(Quadrilateral, (20, 20))
        f_s = x ->  1.0 + x[1] + x[2] + x[1] * x[2]
        f_v = x -> Vec{2}((1.0 + x[1] + x[2] + x[1] * x[2], 2.0 - x[1] - x[2] - x[1] * x[2]))
        points = [Vec((x, x, )) for x in range(0; stop=1, length=100)]
        ip_f = Lagrange{RefQuadrilateral,2}()
        ip_f_v = ip_f^2
        qr = QuadratureRule{RefQuadrilateral}(3)       
    end
   
    csv = CellValues(qr, ip_f)
    cvv = CellValues(qr, ip_f_v)
    dh = DofHandler(mesh);
    add!(dh, :s, ip_f)
    add!(dh, :v, ip_f_v)
    close!(dh)
    M = create_matrix(dh)
    f = zeros(ndofs(dh))
    asm = start_assemble(M, f)
    me = zeros(ndofs_per_cell(dh), ndofs_per_cell(dh))
    fe = zeros(ndofs_per_cell(dh))
    s_dofs = dof_range(dh, :s)
    v_dofs = dof_range(dh, :v)
    
    for cell in CellIterator(dh)
        fill!(me, 0)
        fill!(fe, 0)
        reinit!(csv, cell)
        reinit!(cvv, cell)
        for qp in 1:getnquadpoints(csv)
            dΩ = getdetJdV(csv, qp)
            x = spatial_coordinate(csv, qp, getcoordinates(cell))
            for i in 1:getnbasefunctions(csv)
                δui = shape_value(csv, qp, i)
                fe[s_dofs[i]] += ( δui * f_s(x) ) * dΩ
                for j in 1:getnbasefunctions(csv)
                    δuj = shape_value(csv, qp, j)
                    me[s_dofs[i], s_dofs[j]] += δui * δuj * dΩ
                end
            end
            for i in 1:getnbasefunctions(cvv)
                δui = shape_value(cvv, qp, i)
                fe[v_dofs[i]] += ( δui ⋅ f_v(x) ) * dΩ
                for j in 1:getnbasefunctions(cvv)
                    δuj = shape_value(cvv, qp, j)
                    me[v_dofs[i], v_dofs[j]] += δui ⋅ δuj * dΩ
                end
            end
        end
        assemble!(asm, celldofs(cell), me, fe)
    end
    uh = M \ f

    ph = PointEvalHandler(mesh, points)
    @test all(x -> x !== nothing, ph.cells)
    psv = PointValues(ip_f)
    pvv = PointValues(ip_f_v)
    for (x, point) in zip(points, PointIterator(ph))
        point === nothing && continue
        # Test scalar field
        reinit!(psv, point)
        @test function_value(psv, uh[celldofs(dh, cellid(point))], s_dofs) ≈
              function_value(psv, uh[celldofs(dh, cellid(point))][s_dofs]) ≈
              f_s(x)
        @test function_gradient(psv, uh[celldofs(dh, cellid(point))], s_dofs) ≈
              function_gradient(psv, uh[celldofs(dh, cellid(point))][s_dofs]) ≈
              Tensors.gradient(f_s, x)
        # Test vector field
        reinit!(pvv, point)
        @test function_value(pvv, uh[celldofs(dh, cellid(point))], v_dofs) ≈
              function_value(pvv, uh[celldofs(dh, cellid(point))][v_dofs]) ≈
              f_v(x)
        @test function_gradient(pvv, uh[celldofs(dh, cellid(point))], v_dofs) ≈
              function_gradient(pvv, uh[celldofs(dh, cellid(point))][v_dofs]) ≈
              Tensors.gradient(f_v, x)
        @test function_symmetric_gradient(pvv, uh[celldofs(dh, cellid(point))], v_dofs) ≈
              function_symmetric_gradient(pvv, uh[celldofs(dh, cellid(point))][v_dofs]) ≈
              symmetric(Tensors.gradient(f_v, x))
    end
end

function mixed_grid()
    ## Mixed grid where not all cells have the same fields 

    # 5_______6
    # |\      | 
    # |   \   |
    # 3______\4
    # |       |
    # |       |
    # 1_______2 

    nodes = [Node((0.0, 0.0)),
            Node((1.0, 0.0)),
            Node((0.0, 1.0)),
            Node((1.0, 1.0)),
            Node((0.0, 2.0)),
            Node((1.0, 2.0))]

    cells = Ferrite.AbstractCell[Quadrilateral((1,2,4,3)),
            Triangle((3,4,6)),
            Triangle((3,6,5))]

    mesh = Grid(cells, nodes)
    addcellset!(mesh, "quads", Set{Int}((1,)))
    addcellset!(mesh, "tris", Set{Int}((2, 3)))

    ip_quad = Lagrange{RefQuadrilateral,1}()
    ip_tri = Lagrange{RefTriangle,1}()

    f(x) = x[1]

    # compute values in quadrature points for quad
    qr = QuadratureRule{RefQuadrilateral}(2)
    cv = CellValues(qr, ip_quad)
    qp_vals_quads = [Vector{Float64}(undef, getnquadpoints(cv)) for cell in getcellset(mesh, "quads")]
    for (local_cellid, global_cellid) in enumerate(getcellset(mesh, "quads"))
        xe = getcoordinates(mesh, global_cellid)
        reinit!(cv, xe)
        for qp in 1:getnquadpoints(cv)
            qp_vals_quads[local_cellid][qp] = f(spatial_coordinate(cv, qp, xe))
        end
    end

    # construct projector 
    projector = L2Projector(ip_quad, mesh; set=getcellset(mesh, "quads"))

    points = [Vec((x, 2x)) for x in range(0.0; stop=1.0, length=10)]

    # first alternative: L2Projection to dofs
    projector_values = project(projector, qp_vals_quads, qr)
    ph = PointEvalHandler(mesh, points)
    vals = evaluate_at_points(ph, projector, projector_values)
    @test vals[1:5] ≈ f.(points[1:5])
    @test all(isnan, vals[6:end])

    # second alternative: assume a vector field :v
    dh = DofHandler(mesh)
    sdh_quad = SubDofHandler(dh, getcellset(mesh, "quads"))
    add!(sdh_quad, :v, ip_quad^2)
    sdh_tri = SubDofHandler(dh, getcellset(mesh, "tris"))
    add!(sdh_tri, :v, ip_tri^2)
    close!(dh)

    dof_vals = [1., 1., 2., 2., 4., 4., 3., 3., 6., 6., 5., 5.]
    points = [node.x for node in mesh.nodes]
    ph = PointEvalHandler(mesh, points)
    vals = evaluate_at_points(ph, dh, dof_vals, :v)
    @test vals == [Vec((i, i)) for i=1.0:6.0]
end

function oneD()
    # isoparametric approximation
    mesh = generate_grid(Line, (2,))
    f(x) = x[1]
    nodal_vals = [f(p.x) for p in mesh.nodes]

    ip_f = Lagrange{RefLine,1}() # function interpolation

    # compute values in quadrature points
    qr = QuadratureRule{RefLine}(2)
    cv = CellValues(qr, ip_f)
    qp_vals = [Vector{Float64}(undef, getnquadpoints(cv)) for i=1:getncells(mesh)]
    for cellid in eachindex(mesh.cells)
        xe = getcoordinates(mesh, cellid)
        reinit!(cv, xe)
        for qp in 1:getnquadpoints(cv)
            qp_vals[cellid][qp] = f(spatial_coordinate(cv, qp, xe))
        end
    end

    # do a L2Projection for getting values in dofs
    projector = L2Projector(ip_f, mesh)
    projector_values = project(projector, qp_vals, qr)

    # points where we want to retrieve field values
    points = [Vec((x,)) for x in range(-1.0; stop=1.0, length=5)]

    # set up PointEvalHandler and retrieve values
    ph = PointEvalHandler(mesh, points)
    vals = evaluate_at_points(ph, projector, projector_values)
    @test f.(points) ≈ vals

    # alternatively retrieve vals from nodal values
    # TODO
    # vals = evaluate_at_points(ph, nodal_vals)
    # @test f.(points) ≈ vals
end

function first_point_missing()
    mesh = generate_grid(Quadrilateral, (1, 1))
    points = [Vec(2.0, 0.0), Vec(0.0, 0.0)]
    ph = PointEvalHandler(mesh, points; warn=false)
    
    @test isnothing(ph.local_coords[1])
    @test ph.local_coords[2] == Vec(0.0, 0.0)
end

@testset "PointEvalHandler" begin
    scalar_field()
    vector_field()
    dofhandler()
    dofhandler2(;three_dimensional=false)
    dofhandler2(;three_dimensional=true)
    superparametric()
    mixed_grid()
    oneD()
    first_point_missing()
end

@testset "PointValues" begin
    ip_f = Lagrange{RefQuadrilateral,2}()
    x = Vec{2,Float64}.([(0.0, 0.0), (2.0, 0.5), (2.5, 2.5), (0.5, 2.0)])
    ξ₁ = Vec{2,Float64}((0.12, -0.34))
    ξ₂ = Vec{2,Float64}((0.56, -0.78))
    qr = QuadratureRule{RefQuadrilateral,Float64}([2.0, 2.0], [ξ₁, ξ₂])

    # PointScalarValues
    csv = CellValues(qr, ip_f)
    reinit!(csv, x)
    psv = PointValues(csv)
    us = rand(getnbasefunctions(ip_f)) .+ 1
    reinit!(psv, x, ξ₁)
    @test function_value(psv, us) ≈ function_value(csv, 1, us)
    @test function_gradient(psv, us) ≈ function_gradient(csv, 1, us)
    reinit!(psv, x, ξ₂)
    @test function_value(psv, us) ≈ function_value(csv, 2, us)
    @test function_gradient(psv, us) ≈ function_gradient(csv, 2, us)

    # PointVectorValues
    cvv = CellValues(qr, ip_f^2)
    reinit!(cvv, x)
    pvv = PointValues(cvv)
    uv = rand(2 * getnbasefunctions(ip_f)) .+ 1
    reinit!(pvv, x, ξ₁)
    @test function_value(pvv, uv) ≈ function_value(cvv, 1, uv)
    @test function_gradient(pvv, uv) ≈ function_gradient(cvv, 1, uv)
    @test function_symmetric_gradient(pvv, uv) ≈ function_symmetric_gradient(cvv, 1, uv)
    reinit!(pvv, x, ξ₂)
    @test function_value(pvv, uv) ≈ function_value(cvv, 2, uv)
    @test function_gradient(pvv, uv) ≈ function_gradient(cvv, 2, uv)
    @test function_symmetric_gradient(pvv, uv) ≈ function_symmetric_gradient(cvv, 2, uv)
end
