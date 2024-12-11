using Ferrite, Tensors, KrylovKit
using Arpack: Arpack

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
    grid = generate_grid(Triangle, (40, 40), zero(Vec{2}), π * ones(Vec{2}))
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
    #Ferrite.zero_out_rows!(B, ch.dofmapping)
    #Ferrite.zero_out_columns!(B, ch.prescribed_dofs)
    fdofs = ch.free_dofs
    return A, B, dh, fdofs
end

ip = Nedelec{2, RefTriangle, 1}()
#ip = Lagrange{RefTriangle, 1}()^2

A, B, dh, fdofs = setup_and_assemble(ip)
Aff = A[fdofs, fdofs]
Bff = B[fdofs, fdofs]

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
