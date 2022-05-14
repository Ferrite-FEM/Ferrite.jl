# # Helmholtz equation
#
# In this example, we want to solve a (variant of) of the [Helmholtz equation](https://en.wikipedia.org/wiki/Helmholtz_equation).
# The example is inspired by an [dealii step_7](https://www.dealii.org/8.4.1/doxygen/deal.II/step_7.html) on the standard square.
#
# ```math
#  - \Delta u + u = f
# ```
#
# With boundary conditions given by
# ```math
# u = g_1 \quad x \in \Gamma_1
# ```
# and
# ```math
# n \cdot \nabla u = g_2 \quad x \in \Gamma_2
# ```
# 
# Here Γ₁ is the union of the top and the right boundary of the square,
# while Γ₂ is the union of the bottom and the left boundary.
#
# ![](helmholtz.png)
#
# We will use the following weak formulation:
# ```math
# \int \nabla δu \cdot \nabla u d\Omega
# + \int δu \cdot u d\Omega
# - \int δu \cdot f d\Omega
# - \int δu g_2 d\Gamma_2 = 0 \forall δu
# ```
#
# where $δu$ is a suitable test function that satisfies:
# ```math
# δu = 0 \quad x \in \Gamma_1
# ```
# and $u$ is a suitable function that satisfies:
# ```math
# u = g_1 \quad x \in \Gamma_1
# ```
# The example highlights the following interesting features:
#
# * There are two kinds of boundary conditions, "Dirichlet" and "Von Neumann"
# * The example contains boundary integrals
# * The Dirichlet condition is imposed strongly and the Von Neumann condition is imposed weakly.
#
using Ferrite
using Tensors
using SparseArrays
using LinearAlgebra

const ∇ = Tensors.gradient
const Δ = Tensors.hessian;

k = 2.0

grid = generate_grid(Quadrilateral, (150, 150))

dim = 2
ip = Lagrange{dim, RefCube, 1}()
qr = QuadratureRule{dim, RefCube}(2)
qr_face = QuadratureRule{dim-1, RefCube}(2)
cellvalues = CellScalarValues(qr, ip);
facevalues = FaceScalarValues(qr_face, ip);

# add Complex?
dh = DofHandler(grid)
push!(dh, :u, 1)
close!(dh)

# We will set things up, so that a known analytic solution is approximately reproduced.
# This is a good testing strategy for PDE codes and known as the method of manufactured solutions.

function u_ana(x::Vec{2, T}) where {T}
    return exp(im*k*x[1])
end;

# The (strong) Dirichlet boundary condition can be handled automatically by the Ferrite library.
dbcs = ConstraintHandler(dh;bctype=ComplexF64)
dbc = Dirichlet(:u, union(getfaceset(grid, "left"), getfaceset(grid, "bottom"), getfaceset(grid, "top"), getfaceset(grid, "right")), (x,t) -> u_ana(x))
add!(dbcs, dbc)
close!(dbcs)
update!(dbcs, 0.0)

K = create_sparsity_pattern(dh);

function doassemble(cellvalues::CellScalarValues{dim}, facevalues::FaceScalarValues{dim},
                         K::SparseMatrixCSC, dh::DofHandler) where {dim}
    b = 1.0
	fill!(K.nzval, zero(ComplexF64))
    f = zeros(ComplexF64, ndofs(dh))
    assembler = start_assemble(K, f)
    
    n_basefuncs = getnbasefunctions(cellvalues)
    global_dofs = zeros(Int, ndofs_per_cell(dh))

    fe = zeros(ComplexF64, n_basefuncs) # Local force vector
    Ke = zeros(ComplexF64, n_basefuncs, n_basefuncs) # Local stiffness mastrix

    @inbounds for (cellcount, cell) in enumerate(CellIterator(dh))
        fill!(Ke, 0)
        fill!(fe, 0)
        coords = getcoordinates(cell)

        reinit!(cellvalues, cell)
        # First we derive the non boundary part of the variation problem from the destined solution `u_ana`
        # ```math
        # \int \nabla δu \cdot \nabla u d\Omega
        # + \int δu \cdot u d\Omega
        # - \int δu \cdot f d\Omega
        # ```
        #+

        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            coords_qp = spatial_coordinate(cellvalues, q_point, coords)
            f_true = zero(ComplexF64)
            for i in 1:n_basefuncs
                δu = shape_value(cellvalues, q_point, i)
                ∇δu = shape_gradient(cellvalues, q_point, i)
                fe[i] += (δu * f_true) * dΩ
                for j in 1:n_basefuncs
                    u = shape_value(cellvalues, q_point, j)
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += (∇δu ⋅ ∇u + δu * u) * dΩ
                end
            end
        end
   
        celldofs!(global_dofs, cell)
        assemble!(assembler, global_dofs, fe, Ke)
    end
    return K, f
end;

# We should be able to remove this at some point
K = create_sparsity_pattern(dh; field_type=ComplexF64)
K, f = doassemble(cellvalues, facevalues, K, dh)
apply!(K, f, dbcs)
u = K \ f
u

vtkfile = vtk_grid("helmholtz", dh)
vtk_point_data(vtkfile, dh, real.(u), "_real")
vtk_point_data(vtkfile, dh, imag.(u), "_imag")
vtk_save(vtkfile)

println("Helmholtz successful")
