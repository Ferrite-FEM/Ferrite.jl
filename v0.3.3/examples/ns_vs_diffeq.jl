using Ferrite, SparseArrays, BlockArrays, LinearAlgebra, UnPack

using OrdinaryDiffEq

ν = 1.0/1000.0; #dynamic viscosity

dim = 2
cell_scale_factor = 2.0
x_cells = round(Int, cell_scale_factor*220)
y_cells = round(Int, cell_scale_factor*41)
x_cells = round(Int, 55/3)                  #hide
y_cells = round(Int, 41/3)                  #hide
grid = generate_grid(Quadrilateral, (x_cells, y_cells), Vec{2}((0.0, 0.0)), Vec{2}((2.2, 0.41)));

cell_indices = filter(ci->norm(mean(map(i->grid.nodes[i].x-[0.2,0.2], Ferrite.vertices(grid.cells[ci]))))>0.05, 1:length(grid.cells))
hole_cell_indices = filter(ci->norm(mean(map(i->grid.nodes[i].x-[0.2,0.2], Ferrite.vertices(grid.cells[ci]))))<=0.05, 1:length(grid.cells));
hole_face_ring = Set{FaceIndex}()
for hci ∈ hole_cell_indices
    push!(hole_face_ring, FaceIndex((hci+1, 4)))
    push!(hole_face_ring, FaceIndex((hci-1, 2)))
    push!(hole_face_ring, FaceIndex((hci-x_cells, 3)))
    push!(hole_face_ring, FaceIndex((hci+x_cells, 1)))
end
grid.facesets["hole"] = Set(filter(x->x.idx[1] ∉ hole_cell_indices, collect(hole_face_ring)));
cell_indices_map = map(ci->norm(mean(map(i->grid.nodes[i].x-[0.2,0.2], Ferrite.vertices(grid.cells[ci]))))>0.05 ? indexin([ci], cell_indices)[1] : 0, 1:length(grid.cells))
grid.cells = grid.cells[cell_indices]
for facesetname in keys(grid.facesets)
    grid.facesets[facesetname] = Set(map(fi -> FaceIndex( cell_indices_map[fi.idx[1]] ,fi.idx[2]), collect(grid.facesets[facesetname])))
end;

grid = generate_grid(Quadrilateral, (x_cells, y_cells), Vec{2}((0.0, 0.0)), Vec{2}((0.55, 0.41)));   #hide

ip_v = Lagrange{dim, RefCube, 2}()
ip_geom = Lagrange{dim, RefCube, 1}()
qr = QuadratureRule{dim, RefCube}(4)
cellvalues_v = CellVectorValues(qr, ip_v, ip_geom);

ip_p = Lagrange{dim, RefCube, 1}()
cellvalues_p = CellScalarValues(qr, ip_p, ip_geom);

dh = DofHandler(grid)
push!(dh, :v, dim, ip_v)
push!(dh, :p, 1, ip_p)
close!(dh);

ch = ConstraintHandler(dh);

nosplip_face_names = ["top", "bottom", "hole"];
nosplip_face_names = ["top", "bottom"]                                  #hide
∂Ω_noslip = union(getfaceset.((grid, ), nosplip_face_names)...);
noslip_bc = Dirichlet(:v, ∂Ω_noslip, (x, t) -> [0,0], [1,2])
add!(ch, noslip_bc);

∂Ω_inflow = getfaceset(grid, "left");

vᵢₙ(t) = clamp(t, 0.0, 1.0)*1.0 #inflow velocity
vᵢₙ(t) = clamp(t, 0.0, 1.0)*0.3 #hide
parabolic_inflow_profile((x,y),t) = [4*vᵢₙ(t)*y*(0.41-y)/0.41^2,0]
inflow_bc = Dirichlet(:v, ∂Ω_inflow, parabolic_inflow_profile, [1,2])
add!(ch, inflow_bc);

∂Ω_free = getfaceset(grid, "right");

close!(ch)
update!(ch, 0.0);

function assemble_mass_matrix(cellvalues_v::CellVectorValues{dim}, cellvalues_p::CellScalarValues{dim}, M::SparseMatrixCSC, dh::DofHandler) where {dim}
    # Allocate a buffer for the local matrix and some helpers, together with the assembler.
    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    n_basefuncs = n_basefuncs_v + n_basefuncs_p
    v▄, p▄ = 1, 2
    Mₑ = PseudoBlockArray(zeros(n_basefuncs, n_basefuncs), [n_basefuncs_v, n_basefuncs_p], [n_basefuncs_v, n_basefuncs_p])

    # It follows the assembly loop as explained in the basic tutorials.
    mass_assembler = start_assemble(M)
    @inbounds for cell in CellIterator(dh)
        fill!(Mₑ, 0)
        Ferrite.reinit!(cellvalues_v, cell)

        for q_point in 1:getnquadpoints(cellvalues_v)
            dΩ = getdetJdV(cellvalues_v, q_point)
            # Remember that we assemble a vector mass term, hence the dot product.
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

function assemble_stokes_matrix(cellvalues_v::CellVectorValues{dim}, cellvalues_p::CellScalarValues{dim}, ν, K::SparseMatrixCSC, dh::DofHandler) where {dim}
    # Again, some buffers and helpers
    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    n_basefuncs = n_basefuncs_v + n_basefuncs_p
    v▄, p▄ = 1, 2
    Kₑ = PseudoBlockArray(zeros(n_basefuncs, n_basefuncs), [n_basefuncs_v, n_basefuncs_p], [n_basefuncs_v, n_basefuncs_p])

    # Assembly loop
    stiffness_assembler = start_assemble(K)
    @inbounds for cell in CellIterator(dh)
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

T = 10.0
Δt₀ = 0.01
Δt_save = 0.1

M = create_sparsity_pattern(dh);
M = assemble_mass_matrix(cellvalues_v, cellvalues_p, M, dh);

K = create_sparsity_pattern(dh);
K = assemble_stokes_matrix(cellvalues_v, cellvalues_p, ν, K, dh);

u₀ = zeros(ndofs(dh))
apply!(u₀, ch);

jac_sparsity = sparse(K);

struct RHSparams
    K::SparseMatrixCSC
    ch::ConstraintHandler
    dh::DofHandler
    cellvalues_v::CellVectorValues
end
p = RHSparams(K, ch, dh, cellvalues_v)

function navierstokes!(du,u_uc,p,t)

    @unpack K,ch,dh,cellvalues_v = p

    u = copy(u_uc)
    update!(ch, t)
    apply!(u, ch)

    # Linear contribution (Stokes operator)
    mul!(du, K, u) # du .= K * u

    # nonlinear contribution
    n_basefuncs = getnbasefunctions(cellvalues_v)
    for cell in CellIterator(dh)
        Ferrite.reinit!(cellvalues_v, cell)
        all_celldofs = celldofs(cell)
        v_celldofs = all_celldofs[dof_range(dh, :v)]
        v_cell = u[v_celldofs]
        for q_point in 1:getnquadpoints(cellvalues_v)
            dΩ = getdetJdV(cellvalues_v, q_point)
            ∇v = function_gradient(cellvalues_v, q_point, v_cell)
            v = function_value(cellvalues_v, q_point, v_cell)
            for j in 1:n_basefuncs
                φⱼ = shape_value(cellvalues_v, q_point, j)

                du[v_celldofs[j]] -= v ⋅ ∇v' ⋅ φⱼ * dΩ
            end
        end
    end

    apply_zero!(du, ch)
end;

rhs = ODEFunction(navierstokes!, mass_matrix=M; jac_prototype=jac_sparsity)
problem = ODEProblem(rhs, u₀, (0.0,T), p);

timestepper = ImplicitEuler()
integrator = init(
    problem, timestepper, initializealg=NoInit(), dt=Δt₀,
    adaptive=true, abstol=1e-3, reltol=1e-3,
    progress=true, progress_steps=1,
    saveat=Δt_save);

pvd = paraview_collection("vortex-street.pvd");
integrator = TimeChoiceIterator(integrator, 0.0:Δt_save:T)
for (u_uc,t) in integrator

    update!(ch, t)
    u = copy(u_uc)
    apply!(u, ch)
    #compress=false flag because otherwise each vtk file will be stored in memory
    vtk_grid("vortex-street-$t.vtu", dh; compress=false) do vtk
        vtk_point_data(vtk,dh,u)
        vtk_save(vtk)
        pvd[t] = vtk
    end
end
vtk_save(pvd);

using Test                                                                  #hide
function compute_divergence(dh, u, cellvalues_v)                            #hide
    divv = 0.0                                                              #hide
    @inbounds for (i,cell) in enumerate(CellIterator(dh))                   #hide
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
@testset "INS OrdinaryDiffEq" begin                                         #hide
    u = copy(integrator.integrator.u)                                       #hide
    apply!(u, ch)                                                           #hide
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

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

