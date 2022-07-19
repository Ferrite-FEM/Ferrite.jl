function get_unitdofvals(dh, fset, field, ncomp=1)
    # Calculate a vector that is one for dofs in `ncomp`
    # direction on `fset`, and zero otherwise. 
    # Note, for scalar fields, ncomp=1 is required
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(field, fset, (x,t)->1.0, ncomp))
    close!(ch)
    update!(ch, 0.0)
    a = zeros(ndofs(dh))
    apply!(a, ch)
    return a
end


function get_lever_arms(dh, fset, field, ncomp=1, rotcomp=2)
    # Calculate lever arms from x0=(0,0,0) for a "force"
    # with normal vector `n` in comp `ncomp`, around axis with 
    # index `rotcomp`
    n = Vec{3}(i-> i==ncomp ? 1.0 : 0.0)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(field, fset, (x,t)->(x×n)[rotcomp], ncomp))
    close!(ch)
    update!(ch, 0.0)
    a = zeros(ndofs(dh))
    apply!(a, ch)
    return a
end

@testset "Neumann" begin
    @testset "DofHandler" begin
        # Setup of test mesh
        Nx = 5; Ny = 5
        grid=generate_grid(Quadrilateral, (Nx, Ny));
        dh=DofHandler(grid); push!(dh, :u, 2); close!(dh);

        # Create Neumann boundary condition
        nh = NeumannHandler(dh)
        fv = FaceVectorValues(
            QuadratureRule{1, RefCube}(2), Lagrange{2, RefCube, 1}())
        fun(_, t, _) = Vec{2}((t, 10t))
        add!(nh, Neumann(:u, grid.facesets["right"], fv, fun))
        
        # Test applying the Neumann bc
        f = zeros(ndofs(dh))
        apply!(f, nh, 1.0)

        # Use the ConstraintHandler to give fixed values on each dof
        # Note half load on node at the end of the edge
        a = zeros(ndofs(dh))
        ch = ConstraintHandler(dh); 
        dbc = Dirichlet(
            :u, grid.facesets["right"], 
            (x,t)-> (abs(x[2])<(1-eps()) ? 1.0 : 0.5)*[1.,10.], [1,2])
        add!(ch, dbc)
        close!(ch)
        update!(ch, 1.0)
        apply!(a, ch)

        @test 2*a/Ny ≈ f   # Side length 2, force distributed per area.
        # 3d
        # Setup of test mesh
        Nx, Ny, Nz = (2,2,2)
        grid=generate_grid(Tetrahedron, (Nx, Ny, Nz));
        dh=DofHandler(grid); 
        push!(dh, :u, 3); push!(dh, :p, 1); close!(dh);
        fset = grid.facesets["right"]
    
        # Create Neumann boundary condition
        nh = NeumannHandler(dh)
        qr = QuadratureRule{2, RefTetrahedron}(2)
        ip = Lagrange{3, RefTetrahedron, 1}()
        fv = FaceVectorValues(qr, ip)
        fv_s = FaceScalarValues(qr, ip)
        x_scale, y_scale, z_scale = rand(3)
        ny = Vec{3}((0,1.,0)); nz = Vec{3}((0,0,1.))
        fun(x,t,n) = t*(x_scale*n + y_scale*ny + z_scale*nz)
        add!(nh, Neumann(:u, fset, fv, fun))
        p_scale = rand()
        add!(nh, Neumann(:p, fset, fv_s, (args...)->p_scale))
        
        # Test applying the Neumann bc
        f = zeros(ndofs(dh))
        apply!(f, nh, 1.0)
    
        #= 
        Due to the triangle mesh, the result will not be evenly 
        distributed nodal forces. Therefore, we can only check 
        that the total load, as well as the moment is correct. 
        We use a Dirichlet to calculate the lever arm for each 
        dof, as well as for identifying which nodes to sum over
        to get total force. 
        =# 
        area = 2*2
        for (i,scale) in enumerate((x_scale, y_scale, z_scale))
            @test area*scale ≈ f ⋅ get_unitdofvals(dh, fset, :u, i)
        end
    
        # Moment wrt. forces in x-direction should be zero 
        lever_arms_x_y = get_lever_arms(dh, fset, :u, 1, 2)
        @test isapprox(0, f ⋅ lever_arms_x_y, atol=1.e-10)
        lever_arms_x_z = get_lever_arms(dh, fset, :u, 1, 3)
        @test isapprox(0, f ⋅ lever_arms_x_z, atol=1.e-10)

        # Moment wrt. forces in y- and z-directions, should be 
        # total force, F=area*scale, times half box width, L=1,
        # when around z- and y-directions. 
        lever_arms_y_z = get_lever_arms(dh, fset, :u, 2, 3)
        @test isapprox(1*area*y_scale, f ⋅ lever_arms_y_z, atol=1.e-10)
        lever_arms_z_y = get_lever_arms(dh, fset, :u, 3, 2)
        @test isapprox(-1*area*z_scale, f ⋅ lever_arms_z_y, atol=1.e-10)

        # Test that the additional field, :p, has gotten its forces 
        # correctly (since we already tested the other directions)
        @test isapprox(area*p_scale, f ⋅ get_unitdofvals(dh, fset, :p), atol=1.e-10)

        # Final check for the full sum to make sure force is 
        # not added where it shouldn't have been. 
        @test sum(f) ≈ area*(x_scale+y_scale+z_scale+p_scale)
    end
end
