struct LinearElasticity{T}
    G::T
    K::T
end

function doassemble(cellvalues::CellScalarValues{dim}, K::SparseMatrixCSC, dh::DofHandler) where {dim}
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)
    @inbounds for cell in CellIterator(dh)
        fill!(Ke, 0)
        fill!(fe, 0)
        reinit!(cellvalues, cell)
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            for i in 1:n_basefuncs
                v  = shape_value(cellvalues, q_point, i)
                ∇v = shape_gradient(cellvalues, q_point, i)
                fe[i] += v * dΩ
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += (∇v ⋅ ∇u) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), fe, Ke)
    end
    return K, f
end

function solve()
    grid = generate_grid(Line, (20,))
    dim = 1
    ip = Lagrange{dim, RefCube, 1}()
    qr = QuadratureRule{dim, RefCube}(2)
    cellvalues = CellScalarValues(qr, ip);
    dh = DofHandler(grid)
    push!(dh, :u, 1)
    close!(dh);
    K = create_sparsity_pattern(dh);
    ch = ConstraintHandler(dh);
    ∂Ω = union(getfaceset.((grid, ), ["left", "right"])...);
    dbc = Dirichlet(:u, ∂Ω, (x, t) -> 0)
    add!(ch, dbc);
    close!(ch)
    JuAFEM.update!(ch, 0.0);
    
    K, f = doassemble(cellvalues, K, dh);
    
    apply!(K, f, ch)
    u = K \ f; 
    return dh, u
end

function create_cook_grid(nx, ny)
    corners = [Tensors.Vec{2}((0.0,   0.0)),
               Tensors.Vec{2}((48.0, 44.0)),
               Tensors.Vec{2}((48.0, 60.0)),
               Tensors.Vec{2}((0.0,  44.0))]
    grid = generate_grid(Triangle, (nx, ny), corners);
    # facesets for boundary conditions
    addfaceset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0);
    addfaceset!(grid, "traction", x -> norm(x[1]) ≈ 48.0);
    return grid
end;

function create_values(interpolation_u, interpolation_p)
    # quadrature rules
    qr      = QuadratureRule{2,RefTetrahedron}(3)
    face_qr = QuadratureRule{1,RefTetrahedron}(3)

    # geometric interpolation
    interpolation_geom = Lagrange{2,RefTetrahedron,1}()

    # cell and facevalues for u
    cellvalues_u = CellVectorValues(qr, interpolation_u, interpolation_geom)
    facevalues_u = FaceVectorValues(face_qr, interpolation_u, interpolation_geom)

    # cellvalues for p
    cellvalues_p = CellScalarValues(qr, interpolation_p, interpolation_geom)

    return cellvalues_u, cellvalues_p, facevalues_u
end;

function create_dofhandler(grid, ipu, ipp)
    dh = DofHandler(grid)
    push!(dh, :u, 2, ipu) # displacement
    push!(dh, :p, 1, ipp) # pressure
    close!(dh)
    return dh
end;

function create_bc(dh)
    dbc = ConstraintHandler(dh)
    add!(dbc, Dirichlet(:u, getfaceset(dh.grid, "clamped"), (x,t) -> zero(Tensors.Vec{2}), [1,2]))
    close!(dbc)
    t = 0.0
    JuAFEM.update!(dbc, t)
    return dbc
end;
        
function doassemble(cellvalues_u::CellVectorValues{dim}, cellvalues_p::CellScalarValues{dim},
                    facevalues_u::FaceVectorValues{dim}, K::SparseMatrixCSC, grid::Grid,
                    dh::DofHandler, mp::LinearElasticity) where {dim}

    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)
    nu = getnbasefunctions(cellvalues_u)
    np = getnbasefunctions(cellvalues_p)

    fe = PseudoBlockArray(zeros(nu + np), [nu, np]) # local force vector
    ke = PseudoBlockArray(zeros(nu + np, nu + np), [nu, np], [nu, np]) # local stiffness matrix

    # traction vector
    t = Tensors.Vec{2}((0.0, 1/16))
    # cache ɛdev outside the element routine to avoid some unnecessary allocations
    ɛdev = [zero(SymmetricTensor{2, dim}) for i in 1:getnbasefunctions(cellvalues_u)]

    for cell in CellIterator(dh)
        fill!(ke, 0)
        fill!(fe, 0)
        assemble_up!(ke, fe, cell, cellvalues_u, cellvalues_p, facevalues_u, grid, mp, ɛdev, t)
        assemble!(assembler, celldofs(cell), fe, ke)
    end

    return K, f
end;

function assemble_up!(Ke, fe, cell, cellvalues_u, cellvalues_p, facevalues_u, grid, mp, ɛdev, t)

    n_basefuncs_u = getnbasefunctions(cellvalues_u)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    u▄, p▄ = 1, 2
    reinit!(cellvalues_u, cell)
    reinit!(cellvalues_p, cell)

    # We only assemble lower half triangle of the stiffness matrix and then symmetrize it.
    @inbounds for q_point in 1:getnquadpoints(cellvalues_u)
        for i in 1:n_basefuncs_u
            ɛdev[i] = dev(symmetric(shape_gradient(cellvalues_u, q_point, i)))
        end
        dΩ = getdetJdV(cellvalues_u, q_point)
        for i in 1:n_basefuncs_u
            divδu = shape_divergence(cellvalues_u, q_point, i)
            δu = shape_value(cellvalues_u, q_point, i)
            for j in 1:i
                Ke[BlockIndex((u▄, u▄), (i, j))] += 2 * mp.G * ɛdev[i] ⊡ ɛdev[j] * dΩ
            end
        end

        for i in 1:n_basefuncs_p
            δp = shape_value(cellvalues_p, q_point, i)
            for j in 1:n_basefuncs_u
                divδu = shape_divergence(cellvalues_u, q_point, j)
                Ke[BlockIndex((p▄, u▄), (i, j))] += -δp * divδu * dΩ
            end
            for j in 1:i
                p = shape_value(cellvalues_p, q_point, j)
                Ke[BlockIndex((p▄, p▄), (i, j))] += - 1/mp.K * δp * p * dΩ
            end

        end
    end

    symmetrize_lower!(Ke)

    # We integrate the Neumann boundary using the facevalues.
    # We loop over all the faces in the cell, then check if the face
    # is in our `"traction"` faceset.
    @inbounds for face in 1:nfaces(cell)
        if onboundary(cell, face) && (cellid(cell), face) ∈ getfaceset(grid, "traction")
            reinit!(facevalues_u, cell, face)
            for q_point in 1:getnquadpoints(facevalues_u)
                dΓ = getdetJdV(facevalues_u, q_point)
                for i in 1:n_basefuncs_u
                    δu = shape_value(facevalues_u, q_point, i)
                    fe[i] += (δu ⋅ t) * dΓ
                end
            end
        end
    end
end

function symmetrize_lower!(K)
    for i in 1:size(K,1)
        for j in i+1:size(K,1)
            K[i,j] = K[j,i]
        end
    end
end;

function solve(ν, interpolation_u, interpolation_p)
    # material
    Emod = 1.
    Gmod = Emod / 2(1 + ν)
    Kmod = Emod * ν / ((1+ν) * (1-2ν))
    mp = LinearElasticity(Gmod, Kmod)

    # grid, dofhandler, boundary condition
    n = 50
    grid = create_cook_grid(n, n)
    dh = create_dofhandler(grid, interpolation_u, interpolation_p)
    dbc = create_bc(dh)

    # cellvalues
    cellvalues_u, cellvalues_p, facevalues_u = create_values(interpolation_u, interpolation_p)

    # assembly and solve
    K = create_sparsity_pattern(dh);
    K, f = doassemble(cellvalues_u, cellvalues_p, facevalues_u, K, grid, dh, mp);
    apply!(K, f, dbc)
    u = Symmetric(K) \ f;
    return dh, u
end

@testset "Makie Plots" begin

    dh, u = solve()
    
    @test let plot_dh_u_1D = false
        try
            plot(dh,u)
            plot_dh_u_1D = true
        catch
            plot_dh_u_1D = false
        end
    end

    @test let plot_scatter_1D = false
        try
            scatter(dh,u)
            plot_scatter_1D = true
        catch
            plot_scatter_1D = false
        end
    end

    @test let plot_lines_1D = false
        try
            lines(dh,u)
            plot_lines_1D = true
        catch
            plot_lines_1D = false
        end
    end

    @test let plot_scatterlines_1D = false
        try
            scatterlines(dh,u)
            plot_scatterlines_1D = true
        catch
            plot_scatterlines_1D = false
        end
    end
    
    linear    = Lagrange{2,RefTetrahedron,1}()
    quadratic = Lagrange{2,RefTetrahedron,2}()
        
    dh1, u1 = solve(0.4999999, linear, linear)
    dh2, u2 = solve(0.4999999, quadratic, linear);
    
    @test let plot_mesh_arrows_2D = false
        try
            scene = mesh(dh1,u1)
            arrows!(dh1,u1)
            plot_mesh_arrows_2D = true
        catch   
            plot_mesh_arrows_2D = false 
        end 
    end
    
    @test let plot_mesh_warped_wireframe_2D = false
        try
            scene = JuAFEM.warp_by_vector(dh1,u1; scale=1.4)
            mesh!(dh1,u1)
            wireframe!(scene[end][1])
            plot_mesh_warped_wireframe_2D = true
        catch
            plot_mesh_warped_wireframe_2D = false
        end
    end
    
    @test let plot_surface_scalar_2D = false
        try 
            surface(dh1,u1,field=2)
            plot_surface_scalar_2D = true
        catch
            plot_surface_scalar_2D = false
        end
    end

    @test let plot_surface_veccomp_2D = false
        try 
            surface(dh1,u1,field=1, process=(x-> x[1]))
            plot_surface_veccomp_2D = true
        catch
            plot_surface_veccomp_2D= false
        end
    end

    @test let plot_surface_arrows!_2D = false
        try 
            arrows(dh1,u1)
            surface!(dh1,u1,field=2)
            plot_surface_arrows!_2D = true
        catch
            plot_surface_arrows!_2D = false
        end
    end
end
