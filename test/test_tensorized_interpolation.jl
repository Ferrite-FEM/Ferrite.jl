# Imports for parallel (isolated) test execution:
using SparseArrays

@testset "TensorizedInterpolation pipeline" begin

    @testset "CellValues correctness ($TB)" for TB in (Tensor{2, 2}, SymmetricTensor{2, 2})
        nc = Tensors.n_components(TB)
        grid = generate_grid(Triangle, (2, 2))
        ip = TensorizedInterpolation{TB}(Lagrange{RefTriangle, 1}())
        qr = QuadratureRule{RefTriangle}(2)
        cv = CellValues(qr, ip)
        @test Ferrite.shape_value_type(cv) == typeof(zero(TB{Float64}))
        @test Ferrite.shape_gradient_type(cv) == Tensor{3, 2, Float64, 8}
        @test_throws ArgumentError CellValues(qr, ip; update_hessians = true)

        dh = DofHandler(grid)
        add!(dh, :σ, ip)
        close!(dh)
        @test ndofs(dh) == nc * getnnodes(grid)
        @test Ferrite.n_components(dh, :σ) == nc

        # A linear tensor field σ(x) = A + B1 x₁ + B2 x₂ is reproduced exactly
        A = rand(TB{Float64})
        B1 = rand(TB{Float64})
        B2 = rand(TB{Float64})
        σfun(x) = A + B1 * x[1] + B2 * x[2]
        e1, e2 = basevec(Vec{2})
        ∇σ_exact = otimes(B1, e1) + otimes(B2, e2)
        a = zeros(ndofs(dh))
        apply_analytical!(a, dh, :σ, σfun)

        for cell in CellIterator(dh)
            reinit!(cv, cell)
            ue = a[celldofs(cell)]
            for qp in 1:getnquadpoints(cv)
                x = spatial_coordinate(cv, qp, getcoordinates(cell))
                @test @inferred(function_value(cv, qp, ue))::TB{Float64} ≈ σfun(x)
                @test @inferred(function_gradient(cv, qp, ue))::Tensor{3, 2, Float64} ≈ ∇σ_exact
                div_exact = Vec{2}((∇σ_exact[1, 1, 1] + ∇σ_exact[1, 2, 2], ∇σ_exact[2, 1, 1] + ∇σ_exact[2, 2, 2]))
                @test @inferred(function_divergence(cv, qp, ue))::Vec{2, Float64} ≈ div_exact
            end
        end
        # No methods for symmetric gradient / curl of tensor-valued fields
        ue = a[celldofs(dh, 1)]
        @test_throws MethodError function_symmetric_gradient(cv, 1, ue)
        @test_throws MethodError function_curl(cv, 1, ue)

        # evaluate_at_grid_nodes and PointEvalHandler
        vals = evaluate_at_grid_nodes(dh, a, :σ)
        for (nodeid, node) in enumerate(getnodes(grid))
            @test vals[nodeid] ≈ σfun(get_node_coordinate(node))
        end
        points = [Vec{2}((0.11, -0.42)), Vec{2}((-0.7, 0.3))]
        ph = PointEvalHandler(grid, points)
        pvals = evaluate_at_points(ph, dh, a, :σ)
        @test pvals ≈ σfun.(points)
    end

    @testset "FacetValues and InterfaceValues" begin
        grid = generate_grid(Triangle, (2, 2))
        ip = TensorizedInterpolation{SymmetricTensor{2, 2}}(Lagrange{RefTriangle, 1}())
        dh = DofHandler(grid)
        add!(dh, :σ, ip)
        close!(dh)
        σfun(x) = SymmetricTensor{2, 2}((x[1], x[1] + x[2], x[2]))
        a = zeros(ndofs(dh))
        apply_analytical!(a, dh, :σ, σfun)

        fqr = FacetQuadratureRule{RefTriangle}(2)
        fv = FacetValues(fqr, ip)
        for fc in FacetIterator(dh, getfacetset(grid, "right"))
            reinit!(fv, fc)
            ue = a[celldofs(fc)]
            for qp in 1:getnquadpoints(fv)
                x = spatial_coordinate(fv, qp, getcoordinates(fc))
                @test function_value(fv, qp, ue) ≈ σfun(x)
                @test function_gradient(fv, qp, ue) isa Tensor{3, 2, Float64}
            end
        end

        iv = InterfaceValues(fqr, ip)
        topology = ExclusiveTopology(grid)
        ninterfaces = 0
        for ic in InterfaceIterator(dh, topology)
            reinit!(iv, ic)
            ue = a[interfacedofs(ic)]
            for qp in 1:getnquadpoints(iv)
                x = spatial_coordinate(iv.here, qp, getcoordinates(ic.a))
                # The field is continuous: zero jump and average equal to the field value
                @test function_value_jump(iv, qp, ue) ≈ zero(SymmetricTensor{2, 2}) atol = 1.0e-14
                @test function_value_average(iv, qp, ue) ≈ σfun(x)
            end
            ninterfaces += 1
        end
        @test ninterfaces > 0
    end

    @testset "Embedded cells (sdim = 3, rdim = 2)" begin
        nodes = [
            Node(Vec{3}((0.0, 0.0, 0.0))), Node(Vec{3}((1.0, 0.0, 0.1))),
            Node(Vec{3}((1.0, 1.0, 0.2))), Node(Vec{3}((0.0, 1.0, 0.1))),
        ]
        grid = Grid([Quadrilateral((1, 2, 3, 4))], nodes)
        qr = QuadratureRule{RefQuadrilateral}(2)
        ip_geo = Lagrange{RefQuadrilateral, 1}()^3
        x = getcoordinates(grid, 1)

        # 3×3 tensor field: physical gradient regularizes to Tensor{3, 3} and has a divergence
        cv33 = CellValues(qr, TensorizedInterpolation{Tensor{2, 3}}(Lagrange{RefQuadrilateral, 1}()), ip_geo)
        reinit!(cv33, x)
        u33 = rand(getnbasefunctions(cv33))
        @test function_value(cv33, 1, u33) isa Tensor{2, 3, Float64}
        @test function_gradient(cv33, 1, u33) isa Tensor{3, 3, Float64}
        @test function_divergence(cv33, 1, u33) isa Vec{3, Float64}

        # 2×2 tensor field: gradient is MixedTensor3{2, 2, 3}; no canonical divergence
        cv22 = CellValues(qr, TensorizedInterpolation{Tensor{2, 2}}(Lagrange{RefQuadrilateral, 1}()), ip_geo)
        reinit!(cv22, x)
        u22 = rand(getnbasefunctions(cv22))
        @test function_gradient(cv22, 1, u22) isa MixedTensor3{2, 2, 3, Float64}
        @test_throws MethodError function_divergence(cv22, 1, u22)

        # On a flat surface (z = 0 plane) a field linear in (x, y) is reproduced exactly,
        # so gradient and divergence can be verified against the analytical values
        nodes_flat = [
            Node(Vec{3}((0.0, 0.0, 0.0))), Node(Vec{3}((1.0, 0.0, 0.0))),
            Node(Vec{3}((1.0, 1.0, 0.0))), Node(Vec{3}((0.0, 1.0, 0.0))),
        ]
        grid_flat = Grid([Quadrilateral((1, 2, 3, 4))], nodes_flat)
        A, B1, B2 = rand(Tensor{2, 3}), rand(Tensor{2, 3}), rand(Tensor{2, 3})
        σfun(x) = A + B1 * x[1] + B2 * x[2]
        # Fill nodal dof values in data-component order
        u_flat = reduce(vcat, [collect(σfun(get_node_coordinate(n)).data) for n in nodes_flat])
        x_flat = getcoordinates(grid_flat, 1)
        reinit!(cv33, x_flat)
        e1, e2 = basevec(Vec{3}, 1), basevec(Vec{3}, 2)
        ∇σ_exact = otimes(B1, e1) + otimes(B2, e2)
        div_exact = Vec{3}(i -> B1[i, 1] + B2[i, 2])
        for qp in 1:getnquadpoints(cv33)
            x_qp = spatial_coordinate(cv33, qp, x_flat)
            @test function_value(cv33, qp, u_flat) ≈ σfun(x_qp)
            @test function_gradient(cv33, qp, u_flat) ≈ ∇σ_exact
            @test function_divergence(cv33, qp, u_flat) ≈ div_exact
        end
    end

    @testset "Dirichlet and PeriodicDirichlet" begin
        grid = generate_grid(Quadrilateral, (2, 2))
        ip = TensorizedInterpolation{SymmetricTensor{2, 2}}(Lagrange{RefQuadrilateral, 1}())
        dh = DofHandler(grid)
        add!(dh, :σ, ip)
        close!(dh)
        σbc(x) = SymmetricTensor{2, 2}((x[1], x[1] * x[2], x[2]))

        ch = ConstraintHandler(dh)
        # Natural tensor-valued return, all components
        add!(ch, Dirichlet(:σ, getfacetset(grid, "left"), σbc))
        # Selected components (xx and yy) in any order; the un-prescribed (2,1) entry of
        # the returned tensor is ignored
        add!(ch, Dirichlet(:σ, getfacetset(grid, "right"), (x, t) -> SymmetricTensor{2, 2}((10.0, -1.0, 20.0)), [(2, 2), (1, 1)]))
        # Tensor-valued return for a single selected component ((1,2) = (2,1) = one dof)
        add!(ch, Dirichlet(:σ, getfacetset(grid, "top"), σbc, [(1, 2)]))
        add!(ch, Dirichlet(:σ, getfacetset(grid, "bottom"), x -> SymmetricTensor{2, 2}((30.0, -1.0, 40.0)), [(1, 1), (2, 2)]))
        close!(ch)
        update!(ch, 0.0)
        a = zeros(ndofs(dh))
        apply!(a, ch)

        vals = evaluate_at_grid_nodes(dh, a, :σ)
        for (nodeid, node) in enumerate(getnodes(grid))
            x = get_node_coordinate(node)
            if x[2] ≈ -1.0 # bottom: xx and yy (added last, wins at corners)
                @test vals[nodeid][1, 1] ≈ 30.0
                @test vals[nodeid][2, 2] ≈ 40.0
                if x[1] ≈ -1.0 # bottom-left corner: xy still from the left condition
                    @test vals[nodeid][1, 2] ≈ σbc(x)[1, 2]
                end
            elseif x[1] ≈ -1.0 # left: all components prescribed
                @test vals[nodeid] ≈ σbc(x)
            elseif x[1] ≈ 1.0 # right: xx and yy prescribed
                @test vals[nodeid][1, 1] ≈ 10.0
                @test vals[nodeid][2, 2] ≈ 20.0
            elseif x[2] ≈ 1.0 # top (excl. corners): xy = yx prescribed by one dof
                @test vals[nodeid][1, 2] ≈ σbc(x)[1, 2]
                @test vals[nodeid][2, 1] ≈ σbc(x)[2, 1]
            end
        end

        # PeriodicDirichlet: inhomogeneous works (through the Dirichlet wrapper) ...
        chp = ConstraintHandler(dh)
        facet_map = collect_periodic_facets(grid, "left", "right", x -> x + Vec{2}((2.0, 0.0)))
        add!(chp, PeriodicDirichlet(:σ, facet_map, σbc))
        close!(chp)
        update!(chp, 0.0)
        # With zero image (right) dofs, the mirror (left) dofs equal the inhomogeneity
        # σbc(x_mirror) - σbc(x_image), verifying the tensor-valued callback normalization
        ap = zeros(ndofs(dh))
        apply!(ap, chp)
        valsp = evaluate_at_grid_nodes(dh, ap, :σ)
        for (nodeid, node) in enumerate(getnodes(grid))
            x = get_node_coordinate(node)
            if x[1] ≈ -1.0
                @test valsp[nodeid] ≈ σbc(x) - σbc(Vec{2}((1.0, x[2])))
            end
        end
        # ... but a rotation matrix is not supported for tensor-valued fields
        chr = ConstraintHandler(dh)
        @test_throws ArgumentError add!(chr, PeriodicDirichlet(:σ, facet_map, rotation_tensor(π / 2)[:, :]))

        # Component validation
        che = ConstraintHandler(dh)
        # Linear indices are not allowed for tensor-valued fields
        @test_throws ErrorException add!(che, Dirichlet(:σ, getfacetset(grid, "left"), σbc, [2]))
        # (1,2) and (2,1) refer to the same component of a symmetric tensor
        @test_throws ErrorException add!(che, Dirichlet(:σ, getfacetset(grid, "left"), σbc, [(1, 2), (2, 1)]))
        # Out of range
        @test_throws ErrorException add!(che, Dirichlet(:σ, getfacetset(grid, "left"), σbc, [(3, 1)]))
        # Return value validation (checked in update!, called by close!)
        for badf in (
                x -> [1.0, 2.0, 3.0],       # collection return not allowed
                x -> Vec{2}((1.0, 2.0)),    # ... nor is a Vec
                x -> 0.0,                   # number return requires a single component
            )
            chb = ConstraintHandler(dh)
            add!(chb, Dirichlet(:σ, getfacetset(grid, "left"), badf))
            @test_throws ErrorException close!(chb)
        end
        # Cartesian components are not allowed for non-tensor fields
        dhv = DofHandler(grid)
        add!(dhv, :u, Lagrange{RefQuadrilateral, 1}()^2)
        close!(dhv)
        chv = ConstraintHandler(dhv)
        @test_throws ErrorException add!(chv, Dirichlet(:u, getfacetset(grid, "left"), x -> zero(Vec{2}), [(1, 1)]))
        # Duplicates and mixed input are rejected already in the constructor
        @test_throws ErrorException Dirichlet(:σ, getfacetset(grid, "left"), σbc, [(1, 1), (1, 1)])
        @test_throws MethodError Dirichlet(:σ, getfacetset(grid, "left"), σbc, [1, (1, 2)])
    end

    @testset "apply_analytical! error path" begin
        grid = generate_grid(Triangle, (1, 1))
        ip = TensorizedInterpolation{SymmetricTensor{2, 2}}(Lagrange{RefTriangle, 1}())
        dh = DofHandler(grid)
        add!(dh, :σ, ip)
        close!(dh)
        a = zeros(ndofs(dh))
        # Non-symmetric tensor cannot be converted to the symmetric value type
        @test_throws InexactError apply_analytical!(a, dh, :σ, x -> Tensor{2, 2}((1.0, 2.0, 3.0, 4.0)))
        # Non-tensor returns are rejected
        @test_throws ErrorException apply_analytical!(a, dh, :σ, x -> (1.0, 2.0, 3.0))
    end

    @testset "VTK export" begin
        grid = generate_grid(Quadrilateral, (2, 2))
        ip = TensorizedInterpolation{SymmetricTensor{2, 2}}(Lagrange{RefQuadrilateral, 1}())
        dh = DofHandler(grid)
        add!(dh, :σ, ip)
        close!(dh)
        σfun(x) = SymmetricTensor{2, 2}((x[1], x[1] * x[2], x[2]))
        a = zeros(ndofs(dh))
        apply_analytical!(a, dh, :σ, σfun)

        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:σ, getfacetset(grid, "left"), x -> 0.0, [(2, 1)]))
        close!(ch)

        # Project the same analytical field: write_solution and write_projection must agree
        qr = QuadratureRule{RefQuadrilateral}(2)
        proj = L2Projector(grid)
        add!(proj, 1:getncells(grid), ip; qr_rhs = qr)
        close!(proj)
        cv_geo = CellValues(qr, Lagrange{RefQuadrilateral, 1}()^2)
        qpdata = map(1:getncells(grid)) do cellid
            coords = getcoordinates(grid, cellid)
            reinit!(cv_geo, coords)
            [σfun(spatial_coordinate(cv_geo, qp, coords)) for qp in 1:getnquadpoints(qr)]
        end
        projected = project(proj, qpdata)
        @test eltype(projected) <: SymmetricTensor{2, 2}

        # The solution field is evaluated at the grid nodes in Voigt order, identically to
        # how projected tensor data is written
        data_sol = Ferrite._evaluate_at_grid_nodes(dh, a, :σ, Val(true))
        @test size(data_sol) == (3, getnnodes(grid))
        for (nodeid, node) in enumerate(getnodes(grid))
            σ = σfun(get_node_coordinate(node))
            @test data_sol[:, nodeid] ≈ [σ[1, 1], σ[2, 2], σ[1, 2]] # Voigt order
        end
        # Component masks in write_constraints are permuted to the same Voigt order:
        # data component 2 (xy) maps to Voigt position 3
        @test Ferrite._voigt_position_of_components(ip) == [1, 3, 2]
        @test Ferrite._voigt_position_of_components(TensorizedInterpolation{Tensor{2, 2}}(Lagrange{RefQuadrilateral, 1}())) == [1, 4, 3, 2]

        mktempdir() do dir
            for discont in (false, true)
                fname = joinpath(dir, "tensor_export_$discont")
                # ascii format so that the written data can be parsed back below
                VTKGridFile(fname, grid; write_discontinuous = discont, append = false, compress = false, ascii = true) do vtk
                    write_solution(vtk, dh, a)
                    write_projection(vtk, proj, projected, "σ_proj")
                    Ferrite.write_constraints(vtk, ch)
                end
                content = read(fname * ".vtu", String)
                function get_array(name)
                    m = match(Regex("<DataArray[^>]*Name=\"" * name * "\"[^>]*>([^<]*)</DataArray>", "s"), content)
                    @assert m !== nothing
                    return m.match, reshape(parse.(Float64, split(strip(m.captures[1]))), 3, :)
                end
                # Voigt component names attached to the solution field, projection, and bc mask
                for dataname in ("σ", "σ_proj", "σ_bc")
                    tag, _ = get_array(dataname)
                    @test occursin("ComponentName0=\"xx\"", tag)
                    @test occursin("ComponentName1=\"yy\"", tag)
                    @test occursin("ComponentName2=\"xy\"", tag)
                end
                # Solution and projection of the same (linear) field agree, both in Voigt order
                _, data_σ = get_array("σ")
                _, data_proj = get_array("σ_proj")
                @test data_σ ≈ data_proj
                # Verify the actual Voigt values and the bc mask placement against the vtk
                # node coordinates from the file (grid nodes when continuous, per-cell
                # duplicated nodes when discontinuous)
                _, points = get_array("Points") # (3, n_vtk_nodes), z = 0
                _, data_bc = get_array("σ_bc")
                @test size(points, 2) == size(data_σ, 2) == size(data_bc, 2)
                @test all(iszero, data_bc[1, :]) && all(iszero, data_bc[2, :])
                @test count(==(1.0), data_bc[3, :]) > 0
                for i in axes(points, 2)
                    x = Vec{2}((points[1, i], points[2, i]))
                    σ = σfun(x)
                    @test data_σ[:, i] ≈ [σ[1, 1], σ[2, 2], σ[1, 2]]
                    # bc mask (data component 2 = xy on "left") in the xy Voigt row (3)
                    @test data_bc[3, i] == (x[1] ≈ -1.0 ? 1.0 : 0.0)
                end
            end
        end
    end

    @testset "SubDofHandler compatibility and sparsity coupling" begin
        grid = generate_grid(Triangle, (2, 2))
        addcellset!(grid, "a", Set(1:4))
        addcellset!(grid, "b", Set(5:8))
        dh = DofHandler(grid)
        sdh_a = SubDofHandler(dh, getcellset(grid, "a"))
        add!(sdh_a, :w, Lagrange{RefTriangle, 1}()^4)
        sdh_b = SubDofHandler(dh, getcellset(grid, "b"))
        # Same number of components (4) but different value kind
        @test_throws ErrorException add!(sdh_b, :w, TensorizedInterpolation{Tensor{2, 2}}(Lagrange{RefTriangle, 1}()))
        add!(sdh_b, :w, Lagrange{RefTriangle, 1}()^4)
        close!(dh)

        # Component-wise coupling for a tensor field: components 1 (xx) and 3 (yy)
        # decoupled means no stored entries between their dofs
        dh2 = DofHandler(grid)
        add!(dh2, :σ, TensorizedInterpolation{SymmetricTensor{2, 2}}(Lagrange{RefTriangle, 1}()))
        close!(dh2)
        coupling = trues(3, 3)
        coupling[1, 3] = coupling[3, 1] = false
        K = allocate_matrix(dh2; coupling)
        K_full = allocate_matrix(dh2)
        @test nnz(K) < nnz(K_full)
        is_stored(A, i, j) = i in view(rowvals(A), nzrange(A, j))
        for cellid in 1:getncells(grid)
            cdofs = celldofs(dh2, cellid)
            for i in 1:3:length(cdofs), j in 1:3:length(cdofs)
                di, dj = cdofs[i], cdofs[j] # component-1 dofs of two nodes
                @test is_stored(K, di, dj + 1)  # (1, 2) coupled
                @test !is_stored(K, di, dj + 2) # (1, 3) decoupled
                @test !is_stored(K, di + 2, dj) # (3, 1) decoupled
            end
        end
        # Component-wise renumbering groups the dofs in one block per component
        renumber!(dh2, DofOrder.ComponentWise())
        nn = getnnodes(grid)
        @test ndofs(dh2) == 3 * nn
        for cellid in 1:getncells(grid)
            cdofs = celldofs(dh2, cellid)
            for (i, d) in pairs(cdofs)
                c = mod1(i, 3) # component of local dof i (component-fastest layout)
                @test (c - 1) * nn < d <= c * nn
            end
        end
    end

    @testset "MultiFieldCellValues and show" begin
        ip = TensorizedInterpolation{SymmetricTensor{2, 2}}(Lagrange{RefTriangle, 1}())
        qr = QuadratureRule{RefTriangle}(2)
        mcv = MultiFieldCellValues(qr, (σ = ip, u = Lagrange{RefTriangle, 1}()^2))
        grid = generate_grid(Triangle, (1, 1))
        reinit!(mcv, getcoordinates(grid, 1))
        @test shape_value(mcv.σ, 1, 1) isa SymmetricTensor{2, 2, Float64}

        cv = CellValues(qr, ip)
        @test occursin("SymmetricTensor{2, 2}", repr("text/plain", cv))
        fv = FacetValues(FacetQuadratureRule{RefTriangle}(2), ip)
        @test occursin("SymmetricTensor{2, 2}", repr("text/plain", fv))

        # The per-field type summary is only printed with multiple SubDofHandlers
        grid2 = generate_grid(Triangle, (2, 2))
        addcellset!(grid2, "a", Set(1:4))
        addcellset!(grid2, "b", Set(5:8))
        dh = DofHandler(grid2)
        sdh_a = SubDofHandler(dh, getcellset(grid2, "a"))
        add!(sdh_a, :σ, ip)
        add!(sdh_a, :u, Lagrange{RefTriangle, 1}()^2)
        sdh_b = SubDofHandler(dh, getcellset(grid2, "b"))
        add!(sdh_b, :u, Lagrange{RefTriangle, 1}()^2)
        close!(dh)
        dh_repr = repr("text/plain", dh)
        @test occursin(":σ, SymmetricTensor{2, 2}", dh_repr)
        @test occursin(":u, Vec{2}", dh_repr)
    end

    @testset "VectorizedInterpolation" begin
        ip = Lagrange{RefTriangle, 1}()
        # VectorizedInterpolation is an alias for TensorizedInterpolation{Vec{vdim}}
        @test VectorizedInterpolation{2} === TensorizedInterpolation{Vec{2}}
        @test ip^2 === VectorizedInterpolation{2}(ip) === TensorizedInterpolation{Vec{2}}(ip)
        @test VectorizedInterpolation(ip) === ip^2 # vectorize to refdim by default
        @test Ferrite.n_components(ip^2) == 2
        @test getnbasefunctions(ip^2) == 2 * getnbasefunctions(ip)
        @test Ferrite.shape_value_type(ip^2, Float64) == Vec{2, Float64}
        @test repr("text/plain", ip^2) == "Lagrange{RefTriangle, 1}()^2"

        # Type hierarchy: VectorInterpolation is the alias TensorInterpolation{Vec{vdim}}
        # and covers both standalone vector interpolations (Nedelec etc.) and vectorized
        # wrappers
        @test Ferrite.VectorInterpolation{2} === Ferrite.TensorInterpolation{Vec{2}}
        @test Nedelec{RefTriangle, 1}() isa Ferrite.VectorInterpolation
        @test Nedelec{RefTriangle, 1}() isa Ferrite.VectorInterpolation{2}
        @test ip^2 isa Ferrite.VectorInterpolation
        @test ip^2 isa Ferrite.VectorInterpolation{2}

        # Values/gradients/hessians of a vectorized interpolation match the scalar base
        # interpolation componentwise
        ipv = Lagrange{RefTriangle, 2}()^2
        ξ = Vec{2}((0.2, 0.3))
        for I in 1:getnbasefunctions(ipv)
            i, c = divrem(I - 1, 2) .+ (1, 1)
            d2N, dN, N = Ferrite.reference_shape_hessian_gradient_and_value(ipv.ip, ξ, i)
            d2Nv, dNv, Nv = Ferrite.reference_shape_hessian_gradient_and_value(ipv, ξ, I)
            @test Nv[c] == N
            @test iszero(Nv[3 - c])
            @test all(dNv[c, j] == dN[j] for j in 1:2)
            @test all(iszero(dNv[3 - c, j]) for j in 1:2)
            @test all(d2Nv[c, j, k] == d2N[j, k] for j in 1:2, k in 1:2)
            @test all(iszero(d2Nv[3 - c, j, k]) for j in 1:2, k in 1:2)
        end
        # Hessians are supported in CellValues for vectorized interpolations (unlike for
        # second order tensor-valued interpolations)
        cv = CellValues(QuadratureRule{RefTriangle}(2), ipv; update_hessians = true)
        grid = generate_grid(Triangle, (1, 1))
        reinit!(cv, getcoordinates(grid, 1))
        @test Ferrite.shape_hessian(cv, 1, 1) isa Tensor{3, 2, Float64}

        # Embedded case (vdim = 3 on rdim = 2): mixed-dimension gradient, matching the
        # scalar base interpolation
        ipe = Lagrange{RefTriangle, 1}()^3
        for I in 1:getnbasefunctions(ipe)
            i, c = divrem(I - 1, 3) .+ (1, 1)
            dN, N = Ferrite.reference_shape_gradient_and_value(ipe.ip, ξ, i)
            dNe, Ne = Ferrite.reference_shape_gradient_and_value(ipe, ξ, I)
            @test Ne isa Vec{3, Float64}
            @test dNe isa Tensors.MixedTensor2{3, 2, Float64}
            @test all(Ne[v] == (v == c ? N : 0.0) for v in 1:3)
            @test all(dNe[v, j] == (v == c ? dN[j] : 0.0) for v in 1:3, j in 1:2)
        end
    end
end
