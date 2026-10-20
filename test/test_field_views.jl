using Ferrite: FieldView

@testset "FieldView" begin
    grid = generate_grid(Quadrilateral, (2, 2))
    # Distort the grid to get non-trivial gradients
    transform_coordinates!(grid, x -> Vec((x[1] + 0.1 * x[2]^2, x[2] + 0.05 * x[1])))
    qr = QuadratureRule{RefQuadrilateral}(3)
    ipu = Lagrange{RefQuadrilateral, 2}()^2
    ipp = Lagrange{RefQuadrilateral, 1}()
    dh = DofHandler(grid)
    add!(dh, :u, ipu)
    add!(dh, :p, ipp)
    close!(dh)
    cv = CellValues(qr, (u = ipu, p = ipp))
    n = ndofs_per_cell(dh)
    ranges = (u = dof_range(dh, :u), p = dof_range(dh, :p))
    a = rand(ndofs(dh))

    @testset "shape and function values" begin
        for cell in CellIterator(dh)
            reinit!(cv, cell)
            ae = a[celldofs(cell)]
            for field in (:u, :p)
                v = FieldView(cv, dh, field)
                fv = getproperty(cv, field)
                r = ranges[field]
                @test getnbasefunctions(v) == n
                @test getnquadpoints(v) == getnquadpoints(cv)
                for q in 1:getnquadpoints(cv)
                    for i in 1:n
                        if i in r
                            j = i - first(r) + 1
                            @test shape_value(v, q, i) == shape_value(fv, q, j)
                            @test shape_gradient(v, q, i) == shape_gradient(fv, q, j)
                        else
                            @test shape_value(v, q, i) === zero(shape_value(fv, q, 1))
                            @test shape_gradient(v, q, i) === zero(shape_gradient(fv, q, 1))
                        end
                    end
                    @test function_value(v, q, ae) ≈ function_value(fv, q, ae, r)
                    @test function_gradient(v, q, ae) ≈ function_gradient(fv, q, ae, r)
                    # The generic evaluation over all element dofs gives the same result
                    @test function_value(v, q, ae, 1:n) ≈ function_value(v, q, ae)
                    @test function_gradient(v, q, ae, 1:n) ≈ function_gradient(v, q, ae)
                end
            end
            vu = FieldView(cv, dh, :u)
            for q in 1:getnquadpoints(cv)
                for i in 1:n
                    @test shape_divergence(vu, q, i) ≈ tr(shape_gradient(vu, q, i))
                    @test shape_symmetric_gradient(vu, q, i) ≈ symmetric(shape_gradient(vu, q, i))
                end
                @test function_divergence(vu, q, ae) ≈ function_divergence(cv.u, q, ae, ranges.u)
                @test function_symmetric_gradient(vu, q, ae) ≈ function_symmetric_gradient(cv.u, q, ae, ranges.u)
                @test function_curl(vu, q, ae) ≈ function_curl(cv.u, q, ae, ranges.u)
            end
        end
    end

    @testset "element assembly with a single loop over the element dofs" begin
        # Stokes-type element matrix, assembled block-wise per field and with field views
        function assemble_blocks!(Ke, cv, ru, rp)
            for q in 1:getnquadpoints(cv)
                dΩ = getdetJdV(cv, q)
                for (i, I) in enumerate(ru)
                    ∇δu = shape_symmetric_gradient(cv.u, q, i)
                    divδu = shape_divergence(cv.u, q, i)
                    for (j, J) in enumerate(ru)
                        Ke[I, J] += (∇δu ⊡ shape_symmetric_gradient(cv.u, q, j)) * dΩ
                    end
                    for (j, J) in enumerate(rp)
                        Ke[I, J] -= divδu * shape_value(cv.p, q, j) * dΩ
                        Ke[J, I] -= divδu * shape_value(cv.p, q, j) * dΩ
                    end
                end
            end
            return Ke
        end
        function assemble_views!(Ke, cv, vu, vp)
            nbf = getnbasefunctions(vu)
            for q in 1:getnquadpoints(cv)
                dΩ = getdetJdV(cv, q)
                for i in 1:nbf, j in 1:nbf
                    Ke[i, j] += (
                        shape_symmetric_gradient(vu, q, i) ⊡ shape_symmetric_gradient(vu, q, j) -
                            shape_divergence(vu, q, i) * shape_value(vp, q, j) -
                            shape_value(vp, q, i) * shape_divergence(vu, q, j)
                    ) * dΩ
                end
            end
            return Ke
        end
        Ke_blocks = zeros(n, n)
        Ke_views = zeros(n, n)
        for cell in CellIterator(dh)
            reinit!(cv, cell)
            fill!(Ke_blocks, 0)
            fill!(Ke_views, 0)
            assemble_blocks!(Ke_blocks, cv, ranges.u, ranges.p)
            vu = FieldView(cv, dh, :u)
            vp = FieldView(cv, dh, :p)
            assemble_views!(Ke_views, cv, vu, vp)
            @test Ke_views ≈ Ke_blocks
            @test (@allocated assemble_views!(Ke_views, cv, vu, vp)) == 0
        end
    end

    @testset "construction" begin
        # From a SubDofHandler, and from separate single-field values
        sdh = only(dh.subdofhandlers)
        cell = first(CellIterator(dh))
        cvu = CellValues(qr, ipu)
        reinit!(cvu, cell)
        reinit!(cv, cell)
        v_sdh = FieldView(cv, sdh, :u)
        v_single = FieldView(cvu, ranges.u, n)
        v_multi = FieldView(cv, dh, :u)
        for q in 1:getnquadpoints(cv), i in 1:n
            @test shape_gradient(v_sdh, q, i) == shape_gradient(v_multi, q, i)
            @test shape_gradient(v_single, q, i) ≈ shape_gradient(v_multi, q, i)
        end
        @test v_sdh.dof_range == ranges.u
        # Inference
        @test (@inferred FieldView(cv.p, ranges.p, n)) isa FieldView
        @inferred shape_value(v_multi, 1, 1)
        @inferred shape_gradient(v_multi, 1, n)
        @inferred function_value(v_multi, 1, zeros(n))
        # Errors
        @test_throws ArgumentError FieldView(cv.u, 1:3, n)
        @test_throws ArgumentError FieldView(cv.p, (n - 2):(n + 1), n)
        @test_throws BoundsError shape_value(v_multi, 1, n + 1)
        @test_throws BoundsError shape_gradient(v_multi, 1, 0)
        @test_throws ArgumentError function_value(v_multi, 1, zeros(n - 1))
    end
end
