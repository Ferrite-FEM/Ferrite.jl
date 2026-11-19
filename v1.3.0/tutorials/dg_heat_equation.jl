using Ferrite, SparseArrays
dim = 2;
grid = generate_grid(Quadrilateral, ntuple(_ -> 20, dim));

topology = ExclusiveTopology(grid);

order = 1;
ip = DiscontinuousLagrange{RefQuadrilateral, order}();
qr = QuadratureRule{RefQuadrilateral}(2);

facet_qr = FacetQuadratureRule{RefQuadrilateral}(2);
cellvalues = CellValues(qr, ip);
facetvalues = FacetValues(facet_qr, ip);
interfacevalues = InterfaceValues(facet_qr, ip);

getdistance(p1::Vec{N, T}, p2::Vec{N, T}) where {N, T} = norm(p1 - p2);
getdiameter(cell_coords::Vector{Vec{N, T}}) where {N, T} = maximum(getdistance.(cell_coords, reshape(cell_coords, (1, :))));

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh);

K = allocate_matrix(dh, topology = topology, interface_coupling = trues(1, 1));

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "right"), (x, t) -> 1.0))
add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> -1.0))
close!(ch);

∂Ωₙ = union(
    getfacetset(grid, "top"),
    getfacetset(grid, "bottom"),
);

function assemble_element!(Ke::Matrix, fe::Vector, cellvalues::CellValues)
    n_basefuncs = getnbasefunctions(cellvalues)
    # Reset to 0
    fill!(Ke, 0)
    fill!(fe, 0)
    # Loop over quadrature points
    for q_point in 1:getnquadpoints(cellvalues)
        # Quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        # Loop over test shape functions
        for i in 1:n_basefuncs
            δu = shape_value(cellvalues, q_point, i)
            ∇δu = shape_gradient(cellvalues, q_point, i)
            # Add contribution to fe
            fe[i] += δu * dΩ
            # Loop over trial shape functions
            for j in 1:n_basefuncs
                ∇u = shape_gradient(cellvalues, q_point, j)
                # Add contribution to Ke
                Ke[i, j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
    return Ke, fe
end

function assemble_interface!(Ki::Matrix, iv::InterfaceValues, μ::Float64)
    # Reset to 0
    fill!(Ki, 0)
    # Loop over quadrature points
    for q_point in 1:getnquadpoints(iv)
        # Get the normal to facet A
        normal = getnormal(iv, q_point)
        # Get the quadrature weight
        dΓ = getdetJdV(iv, q_point)
        # Loop over test shape functions
        for i in 1:getnbasefunctions(iv)
            # Multiply the jump by the negative normal to get the definition from the theory section.
            δu_jump = shape_value_jump(iv, q_point, i) * (-normal)
            ∇δu_avg = shape_gradient_average(iv, q_point, i)
            # Loop over trial shape functions
            for j in 1:getnbasefunctions(iv)
                # Multiply the jump by the negative normal to get the definition from the theory section.
                u_jump = shape_value_jump(iv, q_point, j) * (-normal)
                ∇u_avg = shape_gradient_average(iv, q_point, j)
                # Add contribution to Ki
                Ki[i, j] += -(δu_jump ⋅ ∇u_avg + ∇δu_avg ⋅ u_jump) * dΓ + μ * (δu_jump ⋅ u_jump) * dΓ
            end
        end
    end
    return Ki
end

function assemble_boundary!(fe::Vector, fv::FacetValues)
    # Reset to 0
    fill!(fe, 0)
    # Loop over quadrature points
    for q_point in 1:getnquadpoints(fv)
        # Get the normal to facet A
        normal = getnormal(fv, q_point)
        # Get the quadrature weight
        ∂Ω = getdetJdV(fv, q_point)
        # Loop over test shape functions
        for i in 1:getnbasefunctions(fv)
            δu = shape_value(fv, q_point, i)
            boundary_flux = normal[2]
            fe[i] = boundary_flux * δu * ∂Ω
        end
    end
    return fe
end

function assemble_global(cellvalues::CellValues, facetvalues::FacetValues, interfacevalues::InterfaceValues, K::SparseMatrixCSC, dh::DofHandler, order::Int, dim::Int)
    # Allocate the element stiffness matrix and element force vector
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    Ki = zeros(n_basefuncs * 2, n_basefuncs * 2)
    # Allocate global force vector f
    f = zeros(ndofs(dh))
    # Create an assembler
    assembler = start_assemble(K, f)
    # Loop over all cells
    for cell in CellIterator(dh)
        # Reinitialize cellvalues for this cell
        reinit!(cellvalues, cell)
        # Compute volume integral contribution
        assemble_element!(Ke, fe, cellvalues)
        # Assemble Ke and fe into K and f
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    # Loop over all interfaces
    for ic in InterfaceIterator(dh)
        # Reinitialize interfacevalues for this interface
        reinit!(interfacevalues, ic)
        # Calculate the characteristic size hₑ as the face diameter
        interfacecoords = ∩(getcoordinates(ic)...)
        hₑ = getdiameter(interfacecoords)
        # Calculate μ
        μ = (1 + order)^dim / hₑ
        # Compute interface surface integrals contribution
        assemble_interface!(Ki, interfacevalues, μ)
        # Assemble Ki into K
        assemble!(assembler, interfacedofs(ic), Ki)
    end
    # Loop over domain boundaries with Neumann boundary conditions
    for fc in FacetIterator(dh, ∂Ωₙ)
        # Reinitialize facetvalues for this boundary facet
        reinit!(facetvalues, fc)
        # Compute boundary facet surface integrals contribution
        assemble_boundary!(fe, facetvalues)
        # Assemble fe into f
        assemble!(f, celldofs(fc), fe)
    end
    return K, f
end
K, f = assemble_global(cellvalues, facetvalues, interfacevalues, K, dh, order, dim);

apply!(K, f, ch)
u = K \ f;
VTKGridFile("dg_heat_equation", dh) do vtk
    write_solution(vtk, dh, u)
end;

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
