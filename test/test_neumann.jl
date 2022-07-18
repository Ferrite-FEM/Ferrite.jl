
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
    end
end