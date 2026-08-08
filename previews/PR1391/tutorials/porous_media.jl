using Ferrite, FerriteMeshParser, Tensors, WriteVTK, Downloads

struct Elastic{T}
    C::SymmetricTensor{4, 2, T, 9}
end
function Elastic(; E = 20.0e3, ν = 0.3)
    G = E / 2(1 + ν)
    K = E / 3(1 - 2ν)
    I2 = one(SymmetricTensor{2, 2})
    I4dev = minorsymmetric(otimesu(I2, I2)) - I2 ⊗ I2 / 3
    return Elastic(2G * I4dev + K * I2 ⊗ I2)
end;

function element_routine!(Ke, re, material::Elastic, cv::CellValues, a, args...)
    n_basefuncs = getnbasefunctions(cv)

    for q_point in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q_point)
        ϵ = function_symmetric_gradient(cv, q_point, a)
        σ = material.C ⊡ ϵ
        for i in 1:n_basefuncs
            δ∇N = shape_symmetric_gradient(cv, q_point, i)
            re[i] += (δ∇N ⊡ σ) * dΩ
            for j in 1:n_basefuncs
                ∇N = shape_symmetric_gradient(cv, q_point, j)
                Ke[i, j] += (δ∇N ⊡ material.C ⊡ ∇N) * dΩ
            end
        end
    end
    return
end;

struct PoroElastic{T}
    elastic::Elastic{T} ## Skeleton stiffness
    k::T     ## Permeability of liquid   [mm^4/(Ns)]
    ϕ::T     ## Porosity                 [-]
    α::T     ## Biot's coefficient       [-]
    β::T     ## Liquid compressibility   [1/MPa]
end
PoroElastic(; elastic, k, ϕ, α, β) = PoroElastic(elastic, k, ϕ, α, β);

function element_routine!(Ke, re, m::PoroElastic, cv::MultiFieldCellValues, a, a_old, Δt, sdh)
    dr_u = dof_range(sdh, :u)
    dr_p = dof_range(sdh, :p)

    C = m.elastic.C ## Elastic stiffness

    # Assemble stiffness and force vectors
    for q_point in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q_point)
        p = function_value(cv.p, q_point, a, dr_p)
        p_old = function_value(cv.p, q_point, a_old, dr_p)
        pdot = (p - p_old) / Δt
        ∇p = function_gradient(cv.p, q_point, a, dr_p)
        ϵ = function_symmetric_gradient(cv.u, q_point, a, dr_u)
        tr_ϵ_old = function_divergence(cv.u, q_point, a_old, dr_u)
        tr_ϵ_dot = (tr(ϵ) - tr_ϵ_old) / Δt
        σ_eff = C ⊡ ϵ
        # Variation of u_i
        for (iᵤ, Iᵤ) in pairs(dr_u)
            ∇δNu = shape_symmetric_gradient(cv.u, q_point, iᵤ)
            div_δNu = shape_divergence(cv.u, q_point, iᵤ)
            re[Iᵤ] += (∇δNu ⊡ σ_eff - div_δNu * p * m.α) * dΩ
            for (jᵤ, Jᵤ) in pairs(dr_u)
                ∇Nu = shape_symmetric_gradient(cv.u, q_point, jᵤ)
                Ke[Iᵤ, Jᵤ] += (∇δNu ⊡ C ⊡ ∇Nu) * dΩ
            end
            for (jₚ, Jₚ) in pairs(dr_p)
                Np = shape_value(cv.p, q_point, jₚ)
                Ke[Iᵤ, Jₚ] -= (div_δNu * m.α * Np) * dΩ
            end
        end
        # Variation of p_i
        for (iₚ, Iₚ) in pairs(dr_p)
            δNp = shape_value(cv.p, q_point, iₚ)
            ∇δNp = shape_gradient(cv.p, q_point, iₚ)
            re[Iₚ] += (δNp * (m.α * tr_ϵ_dot + m.β * pdot) + m.k * (∇δNp ⋅ ∇p)) * dΩ
            for (jᵤ, Jᵤ) in pairs(dr_u)
                div_Nu = shape_divergence(cv.u, q_point, jᵤ)
                Ke[Iₚ, Jᵤ] += δNp * (m.α / Δt) * div_Nu * dΩ
            end
            for (jₚ, Jₚ) in pairs(dr_p)
                ∇Np = shape_gradient(cv.p, q_point, jₚ)
                Np = shape_value(cv.p, q_point, jₚ)
                Ke[Iₚ, Jₚ] += (δNp * m.β * Np / Δt + m.k * (∇δNp ⋅ ∇Np)) * dΩ
            end
        end
    end
    return
end;

struct FEDomain{M, CV, SDH <: SubDofHandler}
    material::M
    cellvalues::CV
    sdh::SDH
end;

function doassemble!(K, r, domains::Vector{<:FEDomain}, a, a_old, Δt)
    assembler = start_assemble(K, r)
    for domain in domains
        doassemble!(assembler, domain, a, a_old, Δt)
    end
    return
end;

function doassemble!(assembler, domain::FEDomain, a, a_old, Δt)
    material = domain.material
    cv = domain.cellvalues
    sdh = domain.sdh
    n = ndofs_per_cell(sdh)
    Ke = zeros(n, n)
    re = zeros(n)
    ae_old = zeros(n)
    ae = zeros(n)
    for cell in CellIterator(sdh)
        # copy values from a to ae
        map!(i -> a[i], ae, celldofs(cell))
        map!(i -> a_old[i], ae_old, celldofs(cell))
        fill!(Ke, 0)
        fill!(re, 0)
        reinit!(cv, cell)
        element_routine!(Ke, re, material, cv, ae, ae_old, Δt, sdh)
        assemble!(assembler, celldofs(cell), Ke, re)
    end
    return
end;

function get_grid()
    # Download the grid if not available already
    gridfile = "porous_media_0p25.inp"
    isfile(gridfile) || Downloads.download(Ferrite.asset_url(gridfile), gridfile)

    # Import grid from abaqus mesh
    grid = get_ferrite_grid(gridfile)

    # Create cellsets for each fieldhandler
    addcellset!(grid, "solid3", intersect(getcellset(grid, "solid"), getcellset(grid, "CPS3")))
    addcellset!(grid, "solid4", intersect(getcellset(grid, "solid"), getcellset(grid, "CPS4R")))
    addcellset!(grid, "porous3", intersect(getcellset(grid, "porous"), getcellset(grid, "CPS3")))
    addcellset!(grid, "porous4", intersect(getcellset(grid, "porous"), getcellset(grid, "CPS4R")))
    return grid
end;

function setup_problem(; t_rise = 0.1, u_max = -0.1)

    grid = get_grid()

    # Define materials
    m_solid = Elastic(; E = 20.0e3, ν = 0.3)
    m_porous = PoroElastic(; elastic = Elastic(; E = 10.0e3, ν = 0.3), β = 1 / 15.0e3, α = 0.9, k = 5.0e-3, ϕ = 0.8)

    # Define interpolations
    ipu_quad = Lagrange{RefQuadrilateral, 2}()^2
    ipu_tri = Lagrange{RefTriangle, 2}()^2
    ipp_quad = Lagrange{RefQuadrilateral, 1}()
    ipp_tri = Lagrange{RefTriangle, 1}()

    # Quadrature rules
    qr_quad = QuadratureRule{RefQuadrilateral}(2)
    qr_tri = QuadratureRule{RefTriangle}(2)

    # CellValues
    cvu_quad = CellValues(qr_quad, ipu_quad)
    cvu_tri = CellValues(qr_tri, ipu_tri)
    cmv_quad = MultiFieldCellValues(qr_quad, (u = ipu_quad, p = ipp_quad))
    cmv_tri = MultiFieldCellValues(qr_tri, (u = ipu_tri, p = ipp_tri))

    # Setup the DofHandler
    dh = DofHandler(grid)
    # Solid quads
    sdh_solid_quad = SubDofHandler(dh, getcellset(grid, "solid4"))
    add!(sdh_solid_quad, :u, ipu_quad)
    # Solid triangles
    sdh_solid_tri = SubDofHandler(dh, getcellset(grid, "solid3"))
    add!(sdh_solid_tri, :u, ipu_tri)
    # Porous quads
    sdh_porous_quad = SubDofHandler(dh, getcellset(grid, "porous4"))
    add!(sdh_porous_quad, :u, ipu_quad)
    add!(sdh_porous_quad, :p, ipp_quad)
    # Porous triangles
    sdh_porous_tri = SubDofHandler(dh, getcellset(grid, "porous3"))
    add!(sdh_porous_tri, :u, ipu_tri)
    add!(sdh_porous_tri, :p, ipp_tri)

    close!(dh)

    # Setup the domains
    domains = [
        FEDomain(m_solid, cvu_quad, sdh_solid_quad),
        FEDomain(m_solid, cvu_tri, sdh_solid_tri),
        FEDomain(m_porous, cmv_quad, sdh_porous_quad),
        FEDomain(m_porous, cmv_tri, sdh_porous_tri),
    ]

    # Boundary conditions
    # Sliding for u, except top which is compressed
    # Sealed for p, except top with prescribed zero pressure
    addfacetset!(dh.grid, "sides", x -> x[1] < 1.0e-6 || x[1] ≈ 5.0)
    addfacetset!(dh.grid, "top", x -> x[2] ≈ 10.0)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> zero(Vec{1}), [2]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "sides"), (x, t) -> zero(Vec{1}), [1]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "top"), (x, t) -> u_max * clamp(t / t_rise, 0, 1), [2]))
    add!(ch, Dirichlet(:p, getfacetset(grid, "top_p"), (x, t) -> 0.0))
    close!(ch)

    return dh, ch, domains
end;

function solve(dh, ch, domains; Δt = 0.025, t_total = 1.0)
    K = allocate_matrix(dh)
    r = zeros(ndofs(dh))
    a = zeros(ndofs(dh))
    a_old = copy(a)
    pvd = paraview_collection("porous_media")
    step = 0
    for t in 0:Δt:t_total
        if t > 0
            update!(ch, t)
            apply!(a, ch)
            doassemble!(K, r, domains, a, a_old, Δt)
            apply_zero!(K, r, ch)
            Δa = -K \ r
            apply_zero!(Δa, ch)
            a .+= Δa
            copyto!(a_old, a)
        end
        step += 1
        VTKGridFile("porous_media_$step", dh) do vtk
            write_solution(vtk, dh, a)
            pvd[t] = vtk
        end
    end
    vtk_save(pvd)
    return a
end;

dh, ch, domains = setup_problem()
a = solve(dh, ch, domains);

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
