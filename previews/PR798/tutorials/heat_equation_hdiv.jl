using Ferrite

function create_grid(ny::Int)
    width = 10.0
    length = 40.0
    center_width = 5.0
    center_length = 20.0
    upper_right = Vec((length / 2, width / 2))
    grid = generate_grid(Triangle, (round(Int, ny * length / width), ny), -upper_right, upper_right)
    addcellset!(grid, "center", x -> abs(x[1]) < center_length / 2 && abs(x[2]) < center_width / 2)
    addcellset!(grid, "around", setdiff(1:getncells(grid), getcellset(grid, "center")))
    return grid
end

grid = create_grid(10)

ip_geo = geometric_interpolation(getcelltype(grid))
ipu = DiscontinuousLagrange{RefTriangle, 0}()
ipq = Ferrite.BrezziDouglasMarini{2, RefTriangle, 1}()
qr = QuadratureRule{RefTriangle}(2)
cellvalues = (u = CellValues(qr, ipu, ip_geo), q = CellValues(qr, ipq, ip_geo))

dh = DofHandler(grid)
add!(dh, :u, ipu)
add!(dh, :q, ipq)
close!(dh)

Γ = union((getfacetset(grid, name) for name in ("left", "right", "bottom", "top"))...)

function assemble_element!(Ke::Matrix, fe::Vector, cv::NamedTuple, dr::NamedTuple, k::Number)
    cvu = cv[:u]
    cvq = cv[:q]
    dru = dr[:u]
    drq = dr[:q]
    h = 1.0 # Heat source
    # Loop over quadrature points
    for q_point in 1:getnquadpoints(cvu)
        # Get the quadrature weight
        dΩ = getdetJdV(cvu, q_point)
        # Loop over test shape functions
        for (iu, Iu) in pairs(dru)
            δNu = shape_value(cvu, q_point, iu)
            # Add contribution to fe
            fe[Iu] += δNu * h * dΩ
            # Loop over trial shape functions
            for (jq, Jq) in pairs(drq)
                div_Nq = shape_divergence(cvq, q_point, jq)
                # Add contribution to Ke
                Ke[Iu, Jq] += (δNu * div_Nq) * dΩ
            end
        end
        for (iq, Iq) in pairs(drq)
            δNq = shape_value(cvq, q_point, iq)
            div_δNq = shape_divergence(cvq, q_point, iq)
            for (ju, Ju) in pairs(dru)
                Nu = shape_value(cvu, q_point, ju)
                Ke[Iq, Ju] -= div_δNq * k * Nu * dΩ
            end
            for (jq, Jq) in pairs(drq)
                Nq = shape_value(cvq, q_point, jq)
                Ke[Iq, Jq] += (δNq ⋅ Nq) * dΩ
            end
        end
    end
    return Ke, fe
end

function assemble_global(cellvalues, dh::DofHandler)
    grid = dh.grid
    # Allocate the element stiffness matrix and element force vector
    dofranges = (u = dof_range(dh, :u), q = dof_range(dh, :q))
    ncelldofs = ndofs_per_cell(dh)
    Ke = zeros(ncelldofs, ncelldofs)
    fe = zeros(ncelldofs)
    # Allocate global system matrix and vector
    K = allocate_matrix(dh)
    f = zeros(ndofs(dh))
    # Create an assembler
    assembler = start_assemble(K, f)
    x = copy(getcoordinates(grid, 1))
    dofs = copy(celldofs(dh, 1))
    # Loop over all cells
    for (cells, k) in (
            (getcellset(grid, "center"), 0.1),
            (getcellset(grid, "around"), 1.0),
        )
        for cellnr in cells
            # Reinitialize cellvalues for this cell
            cell = getcells(grid, cellnr)
            getcoordinates!(x, grid, cell)
            celldofs!(dofs, dh, cellnr)
            reinit!(cellvalues[:u], cell, x)
            reinit!(cellvalues[:q], cell, x)
            # Reset to 0
            fill!(Ke, 0)
            fill!(fe, 0)
            # Compute element contribution
            assemble_element!(Ke, fe, cellvalues, dofranges, k)
            # Assemble Ke and fe into K and f
            assemble!(assembler, dofs, Ke, fe)
        end
    end
    return K, f
end

K, f = assemble_global(cellvalues, dh);
u = K \ f

temperature_dof = first(dof_range(dh, :u))
u_cells = map(1:getncells(grid)) do i
    u[celldofs(dh, i)[temperature_dof]]
end
VTKGridFile("heat_equation_hdiv", dh) do vtk
    write_cell_data(vtk, u_cells, "temperature")
end

function calculate_flux(dh, boundary_facets, ip, a)
    grid = dh.grid
    qr = FacetQuadratureRule{RefTriangle}(4)
    ip_geo = geometric_interpolation(getcelltype(grid))
    fv = FacetValues(qr, ip, ip_geo)

    dofrange = dof_range(dh, :q)
    flux = 0.0
    dofs = celldofs(dh, 1)
    ae = zeros(length(dofs))
    x = getcoordinates(grid, 1)
    for (cellnr, facetnr) in boundary_facets
        getcoordinates!(x, grid, cellnr)
        cell = getcells(grid, cellnr)
        celldofs!(dofs, dh, cellnr)
        map!(i -> a[i], ae, dofs)
        reinit!(fv, cell, x, facetnr)
        for q_point in 1:getnquadpoints(fv)
            dΓ = getdetJdV(fv, q_point)
            n = getnormal(fv, q_point)
            q = function_value(fv, q_point, ae, dofrange)
            flux += (q ⋅ n) * dΓ
        end
    end
    return flux
end

println("Outward flux: ", calculate_flux(dh, Γ, ipq, u))

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
