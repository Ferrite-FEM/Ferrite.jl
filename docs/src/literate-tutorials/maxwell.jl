#=
# Solving Maxwell's equations
```math
\begin{align*}
\nabla \cdot \boldsymbol{D} &= \rho_\mathrm{f} \\
\nabla \times \boldsymbol{H} &= \boldsymbol{J}_\mathrm{f} + \frac{\partial \boldsymbol{D}}{\partial t} \\
\nabla \cdot \boldsymbol{B} &= 0 \\
\nabla \times \boldsymbol{E} &= -\frac{\partial \boldsymbol{B}}{\partial t}
\end{align*}
```
=#

#=
# Maxwell eigenvalue problem
Strong form
```math
\begin{align*}
\mathrm{curl}(\mathrm{curl}(\boldsymbol{u})) &= \lambda \boldsymbol{u} \text{ in } \Omega \\
\boldsymbol{u} \times \boldsymbol{n} &= \boldsymbol{0} \text{ on } \partial\Omega
\end{align*}
```
Weak form
```math
\int_\Omega \mathrm{curl}(\boldsymbol{\delta u}) \cdot \mathrm{curl}(\boldsymbol{u})\ \mathrm{d}\Omega = \lambda \int_\Omega \boldsymbol{\delta u} \cdot \boldsymbol{u}\ \mathrm{d}\Omega \quad \forall\ \boldsymbol{\delta u} \in H_0(\text{curl})
```
Finite element formulation
```math
\underbrace{\int_\Omega \mathrm{curl}(\boldsymbol{\delta N}_i) \cdot \mathrm{curl}(\boldsymbol{N_j})\ \mathrm{d}\Omega}_{A_{ij}}\ x_j = \lambda \underbrace{\int_\Omega \boldsymbol{\delta N_i} \cdot \boldsymbol{N_j}\ \mathrm{d}\Omega}_{B_{ij}}\ x_j
```
=#

# ## Implementation
using Ferrite, Tensors, KrylovKit
using Arpack: Arpack
# ### Element routine
function element_routine!(Ae, Be, cv::CellValues)
    for q_point in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q_point)
        for i in 1:getnbasefunctions(cv)
            δN = shape_value(cv, q_point, i)
            curl_δN = shape_curl(cv, q_point, i)
            for j in 1:getnbasefunctions(cv)
                N = shape_value(cv, q_point, j)
                curl_N = shape_curl(cv, q_point, j)
                Ae[i, j] = (curl_δN ⋅ curl_N) * dΩ
                Be[i, j] = (δN ⋅ N) * dΩ
            end
        end
    end
    return
end

# ### FE setup
function doassemble!(A, B, dh, cv)
    n = ndofs_per_cell(dh)
    Ae = zeros(n, n)
    Be = zeros(n, n)
    a_assem, b_assem = start_assemble.((A, B))
    for cc in CellIterator(dh)
        cell = getcells(dh.grid, cellid(cc))
        reinit!(cv, cell, getcoordinates(cc))
        element_routine!(Ae, Be, cv)
        assemble!(a_assem, celldofs(cc), Ae)
        assemble!(b_assem, celldofs(cc), Be)
    end
    return A, B
end

function setup_and_assemble(ip::VectorInterpolation{2, RefTriangle})
    grid = generate_grid(Triangle, (10, 10))
    cv = CellValues(QuadratureRule{RefTriangle}(2), ip, geometric_interpolation(Triangle))
    dh = close!(add!(DofHandler(grid), :u, ip))
    ∂Ω = union((getfacetset(grid, k) for k in ("left", "top", "right", "bottom"))...)
    dbc = Dirichlet(:u, ∂Ω, ip isa VectorizedInterpolation ? Returns([0.0, 0.0]) : Returns(0.0))
    ch = close!(add!(ConstraintHandler(dh), dbc))
    sp = init_sparsity_pattern(dh)
    add_sparsity_entries!(sp, dh)
    A = allocate_matrix(sp)
    B = allocate_matrix(sp)
    doassemble!(A, B, dh, cv)
    Ferrite.zero_out_rows!(B, ch.dofmapping)
    Ferrite.zero_out_columns!(B, ch.prescribed_dofs)
    return A, B, dh
end

ip = Nedelec{2, RefTriangle, 1}()

A, B, dh = setup_and_assemble(ip)

# vals, vecs, info = geneigsolve((A, B), 1, EigSorter(x -> abs(x - 5.0)); maxiter = 1000);
# λ, ϕ = Arpack.eigs(A, B, nev = 2, sigma=5.5);
