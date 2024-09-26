 if isdefined(Main, :is_ci) #hide
     IS_CI = Main.is_ci     #hide
 else                       #hide
     IS_CI = false          #hide
 end                        #hide
 nothing                    #hide

using Ferrite, SparseArrays, BlockArrays, LinearAlgebra, UnPack, LinearSolve, WriteVTK

using OrdinaryDiffEq

ν = 1.0/1000.0; #dynamic viscosity

using FerriteGmsh
using FerriteGmsh: Gmsh
Gmsh.initialize()
gmsh.option.set_number("General.Verbosity", 2)
dim = 2;

if !IS_CI                                                                                           #hide
rect_tag = gmsh.model.occ.add_rectangle(0, 0, 0, 1.1, 0.41)
circle_tag = gmsh.model.occ.add_circle(0.2, 0.2, 0, 0.05)
circle_curve_tag = gmsh.model.occ.add_curve_loop([circle_tag])
circle_surf_tag = gmsh.model.occ.add_plane_surface([circle_curve_tag])
gmsh.model.occ.cut([(dim,rect_tag)],[(dim,circle_surf_tag)]);
else                                                                                                #hide
rect_tag = gmsh.model.occ.add_rectangle(0, 0, 0, 0.55, 0.41);                                       #hide
end                                                                                                 #hide
nothing                                                                                             #hide

gmsh.model.occ.synchronize()

if !IS_CI                                                                                           #hide
bottomtag = gmsh.model.model.add_physical_group(dim-1,[6],-1,"bottom")
lefttag = gmsh.model.model.add_physical_group(dim-1,[7],-1,"left")
righttag = gmsh.model.model.add_physical_group(dim-1,[8],-1,"right")
toptag = gmsh.model.model.add_physical_group(dim-1,[9],-1,"top")
holetag = gmsh.model.model.add_physical_group(dim-1,[5],-1,"hole");
else                                                                                                #hide
gmsh.model.model.add_physical_group(dim-1,[4],7,"left")                                             #hide
gmsh.model.model.add_physical_group(dim-1,[3],8,"top")                                              #hide
gmsh.model.model.add_physical_group(dim-1,[2],9,"right")                                            #hide
gmsh.model.model.add_physical_group(dim-1,[1],10,"bottom");                                         #hide
end #hide
nothing                                                                                             #hide

gmsh.option.setNumber("Mesh.Algorithm",11)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",20)
gmsh.option.setNumber("Mesh.MeshSizeMax",0.05)
if IS_CI                                                                                            #hide
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",20)                                              #hide
gmsh.option.setNumber("Mesh.MeshSizeMax",0.15)                                                      #hide
end                                                                                                 #hide

gmsh.model.mesh.generate(dim)
grid = togrid()
Gmsh.finalize();

ip_v = Lagrange{RefQuadrilateral, 2}()^dim
qr = QuadratureRule{RefQuadrilateral}(4)
cellvalues_v = CellValues(qr, ip_v);

ip_p = Lagrange{RefQuadrilateral, 1}()
cellvalues_p = CellValues(qr, ip_p);

dh = DofHandler(grid)
add!(dh, :v, ip_v)
add!(dh, :p, ip_p)
close!(dh);

ch = ConstraintHandler(dh);

nosplip_facet_names = ["top", "bottom", "hole"];
if IS_CI                                                                #hide
nosplip_facet_names = ["top", "bottom"]                                 #hide
end                                                                     #hide
∂Ω_noslip = union(getfacetset.((grid, ), nosplip_facet_names)...);
noslip_bc = Dirichlet(:v, ∂Ω_noslip, (x, t) -> Vec((0.0,0.0)), [1,2])
add!(ch, noslip_bc);

∂Ω_inflow = getfacetset(grid, "left");

vᵢₙ(t) = min(t*1.5, 1.5) #inflow velocity

parabolic_inflow_profile(x,t) = Vec((4*vᵢₙ(t)*x[2]*(0.41-x[2])/0.41^2, 0.0))
inflow_bc = Dirichlet(:v, ∂Ω_inflow, parabolic_inflow_profile, [1,2])
add!(ch, inflow_bc);

∂Ω_free = getfacetset(grid, "right");

close!(ch)
update!(ch, 0.0);

function assemble_mass_matrix(cellvalues_v::CellValues, cellvalues_p::CellValues, M::SparseMatrixCSC, dh::DofHandler)
    # Allocate a buffer for the local matrix and some helpers, together with the assembler.
    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    n_basefuncs = n_basefuncs_v + n_basefuncs_p
    v▄, p▄ = 1, 2
    Mₑ = BlockedArray(zeros(n_basefuncs, n_basefuncs), [n_basefuncs_v, n_basefuncs_p], [n_basefuncs_v, n_basefuncs_p])

    # It follows the assembly loop as explained in the basic tutorials.
    mass_assembler = start_assemble(M)
    for cell in CellIterator(dh)
        fill!(Mₑ, 0)
        Ferrite.reinit!(cellvalues_v, cell)

        for q_point in 1:getnquadpoints(cellvalues_v)
            dΩ = getdetJdV(cellvalues_v, q_point)
            # Remember that we assemble a vector mass term, hence the dot product.
            # There is only one time derivative on the left hand side, so only one mass block is non-zero.
            for i in 1:n_basefuncs_v
                φᵢ = shape_value(cellvalues_v, q_point, i)
                for j in 1:n_basefuncs_v
                    φⱼ = shape_value(cellvalues_v, q_point, j)
                    Mₑ[BlockIndex((v▄, v▄), (i, j))] += φᵢ ⋅ φⱼ * dΩ
                end
            end
        end
        assemble!(mass_assembler, celldofs(cell), Mₑ)
    end

    return M
end;

function assemble_stokes_matrix(cellvalues_v::CellValues, cellvalues_p::CellValues, ν, K::SparseMatrixCSC, dh::DofHandler)
    # Again, some buffers and helpers
    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    n_basefuncs = n_basefuncs_v + n_basefuncs_p
    v▄, p▄ = 1, 2
    Kₑ = BlockedArray(zeros(n_basefuncs, n_basefuncs), [n_basefuncs_v, n_basefuncs_p], [n_basefuncs_v, n_basefuncs_p])

    # Assembly loop
    stiffness_assembler = start_assemble(K)
    for cell in CellIterator(dh)
        # Don't forget to initialize everything
        fill!(Kₑ, 0)

        Ferrite.reinit!(cellvalues_v, cell)
        Ferrite.reinit!(cellvalues_p, cell)

        for q_point in 1:getnquadpoints(cellvalues_v)
            dΩ = getdetJdV(cellvalues_v, q_point)

            for i in 1:n_basefuncs_v
                ∇φᵢ = shape_gradient(cellvalues_v, q_point, i)
                for j in 1:n_basefuncs_v
                    ∇φⱼ = shape_gradient(cellvalues_v, q_point, j)
                    Kₑ[BlockIndex((v▄, v▄), (i, j))] -= ν * ∇φᵢ ⊡ ∇φⱼ * dΩ
                end
            end

            for j in 1:n_basefuncs_p
                ψ = shape_value(cellvalues_p, q_point, j)
                for i in 1:n_basefuncs_v
                    divφ = shape_divergence(cellvalues_v, q_point, i)
                    Kₑ[BlockIndex((v▄, p▄), (i, j))] += (divφ * ψ) * dΩ
                    Kₑ[BlockIndex((p▄, v▄), (j, i))] += (ψ * divφ) * dΩ
                end
            end
        end

        # Assemble `Kₑ` into the Stokes matrix `K`.
        assemble!(stiffness_assembler, celldofs(cell), Kₑ)
    end
    return K
end;

T = 6.0
Δt₀ = 0.001
if IS_CI                                                                #hide
    Δt₀ = 0.1                                                           #hide
end                                                                     #hide
Δt_save = 0.1

M = allocate_matrix(dh);
M = assemble_mass_matrix(cellvalues_v, cellvalues_p, M, dh);

K = allocate_matrix(dh);
K = assemble_stokes_matrix(cellvalues_v, cellvalues_p, ν, K, dh);

u₀ = zeros(ndofs(dh))
apply!(u₀, ch);

jac_sparsity = sparse(K);

apply!(M, ch)

struct RHSparams
    K::SparseMatrixCSC
    ch::ConstraintHandler
    dh::DofHandler
    cellvalues_v::CellValues
    u::Vector
end
p = RHSparams(K, ch, dh, cellvalues_v, copy(u₀))

function ferrite_limiter!(u, _, p, t)
    update!(p.ch, t)
    apply!(u, p.ch)
end

function navierstokes_rhs_element!(dvₑ, vₑ, cellvalues_v)
    n_basefuncs = getnbasefunctions(cellvalues_v)
    for q_point in 1:getnquadpoints(cellvalues_v)
        dΩ = getdetJdV(cellvalues_v, q_point)
        ∇v = function_gradient(cellvalues_v, q_point, vₑ)
        v = function_value(cellvalues_v, q_point, vₑ)
        for j in 1:n_basefuncs
            φⱼ = shape_value(cellvalues_v, q_point, j)

            dvₑ[j] -= v ⋅ ∇v' ⋅ φⱼ * dΩ
        end
    end
end

function navierstokes!(du,u_uc,p::RHSparams,t)

    @unpack K,ch,dh,cellvalues_v,u = p

    u .= u_uc
    update!(ch, t)
    apply!(u, ch)

    # Linear contribution (Stokes operator)
    mul!(du, K, u) # du .= K * u

    # nonlinear contribution
    v_range = dof_range(dh, :v)
    n_basefuncs = getnbasefunctions(cellvalues_v)
    vₑ = zeros(n_basefuncs)
    duₑ = zeros(n_basefuncs)
    for cell in CellIterator(dh)
        Ferrite.reinit!(cellvalues_v, cell)
        v_celldofs = @view celldofs(cell)[v_range]
        vₑ .= @views u[v_celldofs]
        fill!(duₑ, 0.0)
        navierstokes_rhs_element!(duₑ, vₑ, cellvalues_v)
        assemble!(du, v_celldofs, duₑ)
    end
end;

function navierstokes_jac_element!(Jₑ, vₑ, cellvalues_v)
    n_basefuncs = getnbasefunctions(cellvalues_v)
    for q_point in 1:getnquadpoints(cellvalues_v)
        dΩ = getdetJdV(cellvalues_v, q_point)
        ∇v = function_gradient(cellvalues_v, q_point, vₑ)
        v = function_value(cellvalues_v, q_point, vₑ)
        for j in 1:n_basefuncs
            φⱼ = shape_value(cellvalues_v, q_point, j)

            for i in 1:n_basefuncs
                φᵢ = shape_value(cellvalues_v, q_point, i)
                ∇φᵢ = shape_gradient(cellvalues_v, q_point, i)
                Jₑ[j, i] -= (φᵢ ⋅ ∇v' + v ⋅ ∇φᵢ') ⋅ φⱼ * dΩ
            end
        end
    end
end

function navierstokes_jac!(J,u_uc,p,t)

    @unpack K, ch, dh, cellvalues_v, u = p

    u .= u_uc
    update!(ch, t)
    apply!(u, ch)

    # Linear contribution (Stokes operator)
    # Here we assume that J has exactly the same structure as K by construction
    nonzeros(J) .= nonzeros(K)

    assembler = start_assemble(J; fillzero=false)

    # Assemble variation of the nonlinear term
    n_basefuncs = getnbasefunctions(cellvalues_v)
    Jₑ = zeros(n_basefuncs, n_basefuncs)
    vₑ = zeros(n_basefuncs)
    v_range = dof_range(dh, :v)
    for cell in CellIterator(dh)
        Ferrite.reinit!(cellvalues_v, cell)
        v_celldofs = @view celldofs(cell)[v_range]

        vₑ .= @views u[v_celldofs]
        fill!(Jₑ, 0.0)
        navierstokes_jac_element!(Jₑ, vₑ, cellvalues_v)
        assemble!(assembler, v_celldofs, Jₑ)
    end

    apply!(J, ch)
end;

rhs = ODEFunction(navierstokes!, mass_matrix=M; jac=navierstokes_jac!, jac_prototype=jac_sparsity)
problem = ODEProblem(rhs, u₀, (0.0,T), p);

struct FreeDofErrorNorm
    ch::ConstraintHandler
end
(fe_norm::FreeDofErrorNorm)(u::Union{AbstractFloat, Complex}, t) = DiffEqBase.ODE_DEFAULT_NORM(u, t)
(fe_norm::FreeDofErrorNorm)(u::AbstractArray, t) = DiffEqBase.ODE_DEFAULT_NORM(u[fe_norm.ch.free_dofs], t)

timestepper = Rodas5P(autodiff=false, step_limiter! = ferrite_limiter!);

integrator = init(
    problem, timestepper; initializealg=NoInit(), dt=Δt₀,
    adaptive=true, abstol=1e-4, reltol=1e-5,
    progress=true, progress_steps=1,
    verbose=true, internalnorm=FreeDofErrorNorm(ch), d_discontinuities=[1.0]
);

pvd = paraview_collection("vortex-street")
for (step, (u,t)) in enumerate(intervals(integrator))
    VTKGridFile("vortex-street-$step", dh) do vtk
        write_solution(vtk, dh, u)
        pvd[t] = vtk
    end
end
vtk_save(pvd);


using Test                                                                      #hide
if IS_CI                                                                        #hide
    function compute_divergence(dh, u, cellvalues_v)                            #hide
        divv = 0.0                                                              #hide
        for cell in CellIterator(dh)                                            #hide
            Ferrite.reinit!(cellvalues_v, cell)                                 #hide
            for q_point in 1:getnquadpoints(cellvalues_v)                       #hide
                dΩ = getdetJdV(cellvalues_v, q_point)                           #hide
                                                                                #hide
                all_celldofs = celldofs(cell)                                   #hide
                v_celldofs = all_celldofs[dof_range(dh, :v)]                    #hide
                v_cell = u[v_celldofs]                                          #hide
                                                                                #hide
                divv += function_divergence(cellvalues_v, q_point, v_cell) * dΩ #hide
            end                                                                 #hide
        end                                                                     #hide
        return divv                                                             #hide
    end                                                                         #hide
    let                                                                         #hide
        u = copy(integrator.u)                                                  #hide
        Δdivv = abs(compute_divergence(dh, u, cellvalues_v))                    #hide
        @test isapprox(Δdivv, 0.0, atol=1e-12)                                  #hide
                                                                                #hide
        Δv = 0.0                                                                #hide
        for cell in CellIterator(dh)                                            #hide
            Ferrite.reinit!(cellvalues_v, cell)                                 #hide
            all_celldofs = celldofs(cell)                                       #hide
            v_celldofs = all_celldofs[dof_range(dh, :v)]                        #hide
            v_cell = u[v_celldofs]                                              #hide
            coords = getcoordinates(cell)                                       #hide
            for q_point in 1:getnquadpoints(cellvalues_v)                       #hide
                dΩ = getdetJdV(cellvalues_v, q_point)                           #hide
                coords_qp = spatial_coordinate(cellvalues_v, q_point, coords)   #hide
                v = function_value(cellvalues_v, q_point, v_cell)               #hide
                Δv += norm(v - parabolic_inflow_profile(coords_qp, T))^2*dΩ     #hide
            end                                                                 #hide
        end                                                                     #hide
        @test isapprox(sqrt(Δv), 0.0, atol=1e-3)                                #hide
    end;                                                                        #hide
    nothing                                                                     #hide
end                                                                             #hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
