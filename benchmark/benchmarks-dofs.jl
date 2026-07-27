#----------------------------------------------------------------------#
# Dof management: dof distribution (close!), renumbering and dof queries
#----------------------------------------------------------------------#
SUITE["dofs"] = BenchmarkGroup()

# Dof distribution. One benchmark per structurally different case rather than a sweep:
#  - vertex dofs only (linear vector field on hexahedra),
#  - vertex + edge + face dofs (quadratic scalar field on tetrahedra),
#  - several fields of different order (Taylor-Hood),
#  - fields on subdomains (two SubDofHandlers with different fields).
SUITE["dofs"]["close!"] = BenchmarkGroup()
let g = SUITE["dofs"]["close!"]
    hexgrid = generate_grid(Hexahedron, (10, 10, 10))
    tetgrid = generate_grid(Tetrahedron, (8, 8, 8))
    quadgrid = generate_grid(Quadrilateral, (40, 40))

    close_singlefield = function (grid, ip)
        dh = DofHandler(grid)
        add!(dh, :u, ip)
        return close!(dh)
    end
    g["Lagrange{1}^3 (Hexahedron 10×10×10)"] = @benchmarkable(
        $close_singlefield($hexgrid, $(Lagrange{RefHexahedron, 1}()^3)), evals = 1
    )
    g["Lagrange{2} (Tetrahedron 8×8×8)"] = @benchmarkable(
        $close_singlefield($tetgrid, $(Lagrange{RefTetrahedron, 2}())), evals = 1
    )

    close_taylorhood = function (grid)
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefQuadrilateral, 2}()^2)
        add!(dh, :p, Lagrange{RefQuadrilateral, 1}())
        return close!(dh)
    end
    g["Taylor-Hood (Quadrilateral 40×40)"] = @benchmarkable $close_taylorhood($quadgrid) evals = 1

    close_subdomains = function (grid)
        dh = DofHandler(grid)
        half = getncells(grid) ÷ 2
        sdh1 = SubDofHandler(dh, Set(1:half))
        add!(sdh1, :u, Lagrange{RefQuadrilateral, 2}()^2)
        add!(sdh1, :p, Lagrange{RefQuadrilateral, 1}())
        sdh2 = SubDofHandler(dh, Set((half + 1):getncells(grid)))
        add!(sdh2, :u, Lagrange{RefQuadrilateral, 2}()^2)
        return close!(dh)
    end
    g["two subdomains (Quadrilateral 40×40)"] = @benchmarkable $close_subdomains($quadgrid) evals = 1
end

# Renumbering and dof queries on a larger Taylor-Hood setup (the celldofs! sweep runs at a
# few ns per cell and needs the cell count to clear the noise floor).
let
    grid = generate_grid(Quadrilateral, (80, 80))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 2}()^2)
    add!(dh, :p, Lagrange{RefQuadrilateral, 1}())
    close!(dh)

    # Recomputing and applying the permutation is the same amount of work whether or not
    # the dofs are already in fieldwise order, so the same handler is reused across samples.
    SUITE["dofs"]["renumber! fieldwise (Taylor-Hood)"] = @benchmarkable renumber!($dh, $(DofOrder.FieldWise())) evals = 1

    dofs = zeros(Int, ndofs_per_cell(dh))
    SUITE["dofs"]["celldofs! all cells (Taylor-Hood)"] = @benchmarkable FerriteBenchmarkHelpers.celldofs_sweep!($dofs, $dh) evals = 1
end
