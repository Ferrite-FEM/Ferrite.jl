# # Bidomain and DifferentialEquations.jl
#

# ```math
# \chi  C_{\textrm{m}} \frac{\partial \varphi_{\textrm{m}}}{\partial t} = \nabla \cdot (\mathbf{\kappa}_{\textrm{i}} \nabla \varphi_{\textrm{m}}) + \nabla \cdot (\mathbf{\kappa}_{\textrm{i}} \nabla \varphi_{\textrm{e}}) - \chi I_{\textrm{ion}}(\varphi_{\textrm{m}}, \mathbf{s}) - \chi I_{\textrm{stim}}(t) \qquad \textrm{on} : \Omega_{\mathbb{H}} \times (0,T]  
# ```
#
# ```math
# 0 = \nabla \cdot (\mathbf{\kappa}_{\textrm{i}} \nabla \varphi_{\textrm{m}}) + \nabla \cdot (\mathbf{\kappa}_e + \mathbf{\kappa}_{\textrm{i}}) \nabla \varphi_{\textrm{e}} \qquad \textrm{on}  : \Omega_{\mathbb{H}} \times (0,T] 
# ```
#
# ```math
# \frac{\partial \mathbf{s}}{\partial t} = \mathbf{g}(\varphi_{\textrm{m}}, \mathbf{s}) \qquad \textrm{on}  : \Omega_{\mathbb{H}} \times (0,T]
# ```
#
# ```math
# f = \varphi_m - 1/3 \varphi_m^3 - s 
# ```
#
# ```math
# g = \varphi_m c - a - b s
# ```
using JuAFEM, SparseArrays, BlockArrays
import DifferentialEquations

grid = generate_grid(Quadrilateral, (20, 20));
dim = 2
ip = Lagrange{dim, RefCube, 1}()
qr = QuadratureRule{dim, RefCube}(2)
cellvalues = CellScalarValues(qr, ip);

dh = DofHandler(grid)
push!(dh, :ϕₘ, 1)
push!(dh, :ϕₑ, 1)
push!(dh, :s, 1)
close!(dh);

K = create_sparsity_pattern(dh);
M = create_sparsity_pattern(dh);

Base.@kwdef struct FHNParameters
    a::Float64 = 0.0
    b::Float64 = 0.8 
    c::Float64 = 1/12.5 
end

function κₑ(x)
    return SymmetricTensor{2,2,Float64}((0.22, 0, 0.13))    
end

function κᵢ(x)
    return SymmetricTensor{2,2,Float64}((0.28, 0, 0.026))
end

function Cₘ(x)
    return 0.01
end

function χ(x)
    return 250.0
end

function Iₛₜᵢₘ(x, t)
    #if norm(x) < 0.2 && t < 2
    #    return 1
    #else
        return 0
    #end
end

ch = ConstraintHandler(dh);
∂Ω = getfaceset(grid, "left");
dbc = Dirichlet(:ϕₑ, ∂Ω, (x, t) -> 0)
add!(ch, dbc);
close!(ch)
update!(ch, 0.0);

function doassemble_linear!(cellvalues::CellScalarValues{dim}, K::SparseMatrixCSC, M::SparseMatrixCSC, dh::DofHandler; params::FHNParameters = FHNParameters()) where {dim}
    n_ϕₘ = getnbasefunctions(cellvalues)
    n_ϕₑ = getnbasefunctions(cellvalues)
    n_s = getnbasefunctions(cellvalues)
    ntotal = n_ϕₘ + n_ϕₑ + n_s
    n_basefuncs = getnbasefunctions(cellvalues)
    
    Ke = PseudoBlockArray(zeros(ntotal, ntotal), [n_ϕₘ, n_ϕₑ, n_s], [n_ϕₘ, n_ϕₑ, n_s])
    Me = PseudoBlockArray(zeros(ntotal, ntotal), [n_ϕₘ, n_ϕₑ, n_s], [n_ϕₘ, n_ϕₑ, n_s])

    assembler_K = start_assemble(K)
    assembler_M = start_assemble(M)

    ϕₘ▄, ϕₑ▄, s▄ = 1, 2, 3

    @inbounds for cell in CellIterator(dh)
        fill!(Ke, 0)
        fill!(Me, 0)
        coords = getcoordinates(cell)

        reinit!(cellvalues, cell)

        for q_point in 1:getnquadpoints(cellvalues)
            coords_qp = spatial_coordinate(cellvalues, q_point, coords)
            κₑ_loc = κₑ(coords_qp)
            κᵢ_loc = κᵢ(coords_qp)
            Cₘ_loc = Cₘ(coords_qp)
            χ_loc = χ(coords_qp)
            dΩ = getdetJdV(cellvalues, q_point)
            for i in 1:n_basefuncs
                v  = shape_value(cellvalues, q_point, i)
                ∇v = shape_gradient(cellvalues, q_point, i)
                for j in 1:n_basefuncs
                    u = shape_value(cellvalues, q_point, j)
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    # diffusion parts
                    Ke[BlockIndex((ϕₘ▄,ϕₘ▄),(i,j))] += ((κᵢ_loc ⋅ ∇v) ⋅ ∇u) * dΩ
                    Ke[BlockIndex((ϕₘ▄,ϕₑ▄),(i,j))] += ((κᵢ_loc ⋅ ∇v) ⋅ ∇u) * dΩ
                    Ke[BlockIndex((ϕₑ▄,ϕₘ▄),(i,j))] += ((κᵢ_loc ⋅ ∇v) ⋅ ∇u) * dΩ
                    Ke[BlockIndex((ϕₑ▄,ϕₑ▄),(i,j))] += (((κₑ_loc + κᵢ_loc) ⋅ ∇v) ⋅ ∇u) * dΩ
                    # linear reaction parts
                    Ke[BlockIndex((ϕₘ▄,ϕₘ▄),(i,j))] += v * u * dΩ 
                    Ke[BlockIndex((ϕₘ▄,s▄),(i,j))] -= v * u * dΩ 
                    Ke[BlockIndex((s▄,ϕₘ▄),(i,j))] += params.c * v * u * dΩ 
                    Ke[BlockIndex((s▄,s▄),(i,j))] -=  params.b * v * u * dΩ 
                    # mass matrices
                    Me[BlockIndex((ϕₘ▄,ϕₘ▄),(i,j))] += Cₘ_loc * χ_loc * v * u * dΩ
                    Me[BlockIndex((s▄,s▄),(i,j))] += v * u * dΩ
                end
            end
        end

        assemble!(assembler_K, celldofs(cell), Ke)
        assemble!(assembler_M, celldofs(cell), Me)
    end
    return K, M
end

K, M = doassemble_linear!(cellvalues, K, M, dh);

apply!(K, ch)
apply!(M, ch)

function bidomain!(du,u,p,t)
    du .= K * u 
end

f = DifferentialEquations.ODEFunction(bidomain!,mass_matrix=M)
u₀ = zeros(ndofs(dh))
prob_mm = DifferentialEquations.ODEProblem(f,u₀,(0.0,2),[K, dh, FHNParameters()])
sol = DifferentialEquations.solve(prob_mm,DifferentialEquations.ImplicitEuler(),reltol=1e-8,abstol=1e-8, progress=true, progress_steps = 1)

