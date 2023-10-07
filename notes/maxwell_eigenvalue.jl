# The Maxwell eigenvalue problem
# Following the Fenics tutorial, 
# [*Stable and unstable finite elements for the Maxwell eigenvalue problem*](https://fenicsproject.org/olddocs/dolfin/2019.1.0/python/demos/maxwell-eigenvalues/demo_maxwell-eigenvalues.py.html),
# we show how Nedelec elements can be used 
# with Ferrite.jl
# ## Problem description
# ### Strong form
# 
# ### Weak form
# ```math
# \int_\Omega \mathrm{curl}(\boldsymbol{\delta u}) \cdot \mathrm{curl}(\boldsymbol{u})\, \mathrm{d}\Omega = \lambda \int_\Omega \boldsymbol{\delta u}\cdot \boldsymbol{u}\ \mathrm{d}\Omega
# ```
# ### FE form 
# ```math
# \begin{align*}
# \int_\Omega \mathrm{curl}(\boldsymbol{\delta N}_i) \cdot \mathrm{curl}(\boldsymbol{N}_j)\, \mathrm{d}\Omega a_j &= \lambda \int_\Omega \boldsymbol{\delta N}_i\cdot \boldsymbol{N}_j\ \mathrm{d}\Omega a_j \\
# A_{ij} a_j &= \lambda B_{ij} a_j
# \end{align*}
# ```

# https://iterativesolvers.julialinearalgebra.org/dev/eigenproblems/lobpcg/
using Ferrite
import Ferrite: Nedelec, RaviartThomas
import IterativeSolvers: lobpcg
using LinearAlgebra
import CairoMakie as M

function assemble_cell!(Ae, Be, cv)
    n = getnbasefunctions(cv)
    for q_point in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q_point)
        for i in 1:n
            δNi = shape_value(cv, q_point, i)
            curl_δNi = shape_curl(cv, q_point, i)
            for j in 1:n
                Nj = shape_value(cv, q_point, j)
                curl_Nj = shape_curl(cv, q_point, j)
                Ae[i,j] += (curl_δNi ⋅ curl_Nj)*dΩ
                Be[i,j] += (δNi ⋅ Nj)*dΩ
            end
        end
    end
    return Ae, Be
end

function plot_shapes(dh, ip, a)
    cv = CellValues(QuadratureRule{RefTriangle}(1), ip, Lagrange{RefTriangle,1}())
    grid = dh.grid
    n_cells = getncells(grid)
    coords = (zeros(n_cells), zeros(n_cells))
    vectors = (zeros(n_cells), zeros(n_cells))

    for cell_nr in 1:getncells(grid)
        x = getcoordinates(grid, cell_nr)
        reinit!(cv, x, getcells(grid, cell_nr))
        ue = a[celldofs(dh, cell_nr)]
        for q_point in 1:getnquadpoints(cv)
            #i = getnquadpoints(cv)*(cell_nr-1) + q_point
            i = cell_nr
            qp_x = spatial_coordinate(cv, q_point, x)
            v = function_value(cv, q_point, ue)
            sfac = norm(v) ≈ 0 ? NaN : 1.0 # Skip plotting zero-vector points
            coords[1][i] = sfac*qp_x[1]
            coords[2][i] = sfac*qp_x[2]
            vectors[1][i] = v[1]
            vectors[2][i] = v[2]
        end
    end
    vtk_grid("tmp", dh.grid) do vtk
        vtk_cell_data(vtk, vectors[1], "u1")
        vtk_cell_data(vtk, vectors[2], "u2")
    end
    nothing, nothing
    #=
    fig = M.Figure()
    for i in 1:2
        ax = M.Axis(fig[i,1]; aspect=M.DataAspect())
        #=for cellnr in 1:getncells(grid)
            x = getcoordinates(grid, cellnr)
            push!(x, x[1])
            M.lines!(ax, first.(x), last.(x), color=:black)
        end=#
        M.scatter!(ax, coords..., vectors[i]; lengthscale=0.1)
    end
    return fig
    =#
end

function doassemble(dh::DofHandler, cv::CellValues)
    grid = dh.grid
    A, B = create_sparsity_pattern.((dh, dh))
    assemA, assemB = start_assemble.((A, B))
    x = getcoordinates(grid, 1)
    n_el_dofs = ndofs_per_cell(dh, 1)
    dofs = zeros(Int, n_el_dofs)
    Ae, Be = [zeros(n_el_dofs, n_el_dofs) for _ in 1:2]

    for (ic, cell) in pairs(getcells(grid))
        getcoordinates!(x, grid, cell)
        celldofs!(dofs, dh, ic)
        reinit!(cv, x, cell)
        fill!.((Ae, Be), 0)
        assemble_cell!(Ae, Be, cv)
        assemble!(assemA, dofs, Ae)
        assemble!(assemB, dofs, Be)
    end
    return A, B
end

function get_matrices(ip::Interpolation; CT=Quadrilateral, nel=40, usebc=true)
    RefShape = Ferrite.getrefshape(ip)
    grid = generate_grid(CT, (nel,nel), zero(Vec{2}), π*ones(Vec{2}))
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)
    ip_geo = Ferrite.default_interpolation(CT)
    cv = CellValues(QuadratureRule{RefShape}(2), ip, ip_geo)
    A, B = doassemble(dh, cv)
    if usebc
        ch = ConstraintHandler(dh)
        dΩh = union(getfaceset(grid, "left"), getfaceset(grid, "right"))
        dΩv = union(getfaceset(grid, "top"), getfaceset(grid, "bottom"))
        if ip isa VectorizedInterpolation
            add!(ch, Dirichlet(:u, dΩh, Returns(0.0), 2)) # y-component on left-right 
            add!(ch, Dirichlet(:u, dΩv, Returns(0.0), 1)) # x-component on top-bottom
        else
            add!(ch, Dirichlet(:u, union!(dΩh,dΩv), Returns(0.0)))
        end
        close!(ch)
        update!(ch, 0.0)
        apply!(A, ch)
        apply!(B, ch)
    end
    return A, B, dh
end

function solve(ip; num_values, kwargs...)
    A, B = get_matrices(ip; kwargs...)
    n = size(A,1)
    r = lobpcg(Symmetric(A), Symmetric(B), false, zeros(n,num_values))
    return r.λ
end

function solve_single(ip, λ=2; kwargs...)
    A, B, dh = get_matrices(ip; kwargs...)
    a = (A-λ*B)\zeros(size(A,1))
    return dh, a
end

ip = Nedelec{2,RefTriangle,1}()
#ip = Lagrange{RefTriangle,1}()^2
dh, a = solve_single(ip, CT=Triangle)
cv = CellValues(QuadratureRule{RefTriangle}(2), ip, Lagrange{RefTriangle,1}())
fig = plot_shapes(dh, ip, a)
#λ = solve(ip; CT=Triangle, num_values=10)

m, n = (1,1)
λ=m^2+n^2
u(x,y)=Vec((sin(m*x),sin(n*y)))
a = zeros(ndofs)
apply_analytical!()