using Ferrite
using SparseArrays

L = 2.0         # Side length
q0 = 10000.0    # Load
E = 200.0e9       # Stiffness
t = 0.01        # Thickness
ν = 0.3         # Poisson's radtio
penalty = 1.0e12  # Penalty stiffness
D = (E * t^3) / (12 * (1 - ν^2)) # Flexural stiffness
C_voigt = D * [
    1.0 ν 0.0;
    ν 1.0 0.0;
    0.0 0.0 (1 - ν) / 2
]
C = fromvoigt(SymmetricTensor{4, 2}, C_voigt)

grid = generate_grid(Triangle, (20, 20), Vec((0.0, 0.0)), Vec((L, L)))

ip = Argyris{RefTriangle, 5}()
dh = DofHandler(grid)
add!(dh, :w, ip)
close!(dh)

qr = QuadratureRule{RefTriangle}(8)
cellvalues = CellValues(qr, ip; update_hessians = true);

fqr = FacetQuadratureRule{RefTriangle}(8)
facetvalues = FacetValues(fqr, ip; update_hessians = true);

function w_analytical(pos::Vec{2}, L, q0, D; n_terms = 50)
    x, y = pos
    w = 0.0
    constant_factor = (16 * q0 * L^4) / (D * pi^6)

    for m in 1:2:n_terms
        for n in 1:2:n_terms
            denom = m * n * (m^2 + n^2)^2
            num = sin(m * pi * x / L) * sin(n * pi * y / L)
            w += num / denom
        end
    end

    return constant_factor * w
end;

function element_routine!(ke, fe, cellvalues, C, q0)
    for iqp in 1:getnquadpoints(cellvalues)
        dV = getdetJdV(cellvalues, iqp)
        for i in 1:getnbasefunctions(cellvalues)
            v = shape_value(cellvalues, iqp, i)
            fe[i] += (q0 * v) * dV
            δκ = shape_hessian(cellvalues, iqp, i)
            for j in 1:getnbasefunctions(cellvalues)
                Δκ = shape_hessian(cellvalues, iqp, j)
                ke[i, j] += (δκ ⊡ C ⊡ Δκ) * dV
            end
        end
    end
    return
end

function bc_routine!(ke, facetvalues, penalty)
    for iqp in 1:getnquadpoints(facetvalues)
        dV = getdetJdV(facetvalues, iqp)
        for i in 1:getnbasefunctions(facetvalues)
            Ni = shape_value(facetvalues, iqp, i)
            for j in 1:getnbasefunctions(facetvalues)
                Nj = shape_value(facetvalues, iqp, j)
                ke[i, j] += penalty * (Ni * Nj) * dV
            end
        end
    end
    return
end

function doassemble!(
        cellvalues::CellValues, facetvalues::FacetValues, K::SparseMatrixCSC, f::Vector, dh::DofHandler, C::SymmetricTensor, q0::Float64, penalty::Float64
    )

    n = getnbasefunctions(cellvalues)
    ke = zeros(n, n)
    fe = zeros(n)

    assembler = start_assemble(K, f)
    for celldata in CellIterator(dh)
        fill!(ke, 0.0)
        fill!(fe, 0.0)
        reinit!(cellvalues, celldata)
        element_routine!(ke, fe, cellvalues, C, q0)
        assemble!(assembler, celldofs(celldata), ke, fe)
    end

    ∂Ω = union(
        getfacetset(grid, "left"),
        getfacetset(grid, "right"),
        getfacetset(grid, "top"),
        getfacetset(grid, "bottom"),
    )

    for celldata in FacetIterator(dh, ∂Ω)
        fill!(ke, 0.0)
        reinit!(facetvalues, celldata)
        bc_routine!(ke, facetvalues, penalty)
        assemble!(assembler, celldofs(celldata), ke)
    end
    return
end

K = allocate_matrix(dh);
f = zeros(ndofs(dh))
doassemble!(cellvalues, facetvalues, K, f, dh, C, q0, penalty);
u = K \ f

VTKGridFile("plate_equation", dh) do vtk
    write_solution(vtk, dh, u)
end

mid_point = Vec((L / 2, L / 2))
ph = PointEvalHandler(grid, [mid_point])
w_fem = evaluate_at_points(ph, dh, u, :w) |> first #0.03548889438239366
w_ana = w_analytical(mid_point, L, q0, D) #0.035488713207468166

using Test
@test w_fem ≈ w_ana atol = 1.0e-6

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
