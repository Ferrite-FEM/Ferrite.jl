function test_apply_rhs()
    grid = generate_grid(Quadrilateral, (20, 20))
    ip = Lagrange{RefQuadrilateral,1}()
    qr = QuadratureRule{RefQuadrilateral}(2)
    cellvalues = CellValues(qr, ip)
    
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)
    
    K = create_matrix(dh)
    
    ch = ConstraintHandler(dh)
    
    ∂Ω = union(getfaceset.((grid,), ["left", "right"])...)
    dbc = Dirichlet(:u, ∂Ω, (x, t) -> 0)
    add!(ch, dbc);

    ∂Ω = union(getfaceset.((grid,), ["top", "bottom"])...)
    dbc = Dirichlet(:u, ∂Ω, (x, t) -> 2)
    add!(ch, dbc);
    
    close!(ch)
    update!(ch, 0.0);
    
    function doassemble!(
        cellvalues::CellValues,
        K::SparseMatrixCSC,
        dh::DofHandler,
    )
    
        n_basefuncs = getnbasefunctions(cellvalues)
        Ke = zeros(n_basefuncs, n_basefuncs)
        fe = zeros(n_basefuncs)
    
        f = zeros(ndofs(dh))
        assembler = start_assemble(K, f)
    
        @inbounds for cell in CellIterator(dh)
            fill!(Ke, 0)
            fill!(fe, 0)
    
            reinit!(cellvalues, cell)
    
            for q_point = 1:getnquadpoints(cellvalues)
                dΩ = getdetJdV(cellvalues, q_point)
    
                for i = 1:n_basefuncs
                    v = shape_value(cellvalues, q_point, i)
                    ∇v = shape_gradient(cellvalues, q_point, i)
                    fe[i] += v * dΩ
                    for j = 1:n_basefuncs
                        ∇u = shape_gradient(cellvalues, q_point, j)
                        Ke[i, j] += (∇v ⋅ ∇u) * dΩ
                    end
                end
            end
    
            assemble!(assembler, celldofs(cell), fe, Ke)
        end
        return K, f
    end
    
    K, f = doassemble!(cellvalues, K, dh)
    A = create_matrix(dh)
    A, g = doassemble!(cellvalues, A, dh)
    rhsdata = get_rhs_data(ch, A)
    
    apply!(K, f, ch)
    apply!(A, ch) # need to apply bcs to A once
    apply_rhs!(rhsdata, g, ch)
    u₁ = K \ f
    u₂ = A \ g
    return u₁, u₂
end
    
u1, u2 = test_apply_rhs()
@test u1 == u2
