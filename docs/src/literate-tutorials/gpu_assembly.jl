# # [Heat equation](@id tutorial-heat-equation)
#
# ![](heat_square.png)
#
# *Figure 1*: Temperature field on the unit square with an internal uniform heat source
# solved with homogeneous Dirichlet boundary conditions on the boundary.
#
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`heat_equation.ipynb`](@__NBVIEWER_ROOT_URL__/examples/heat_equation.ipynb).
#-
#
# ## Introduction
#
# The heat equation is the "Hello, world!" equation of finite elements.
# Here we solve the equation on a unit square, with a uniform internal source.
# The strong form of the (linear) heat equation is given by
#
# ```math
#  -\nabla \cdot (k \nabla u) = f  \quad \textbf{x} \in \Omega,
# ```
#
# where $u$ is the unknown temperature field, $k$ the heat conductivity,
# $f$ the heat source and $\Omega$ the domain. For simplicity we set $f = 1$
# and $k = 1$. We will consider homogeneous Dirichlet boundary conditions such that
# ```math
# u(\textbf{x}) = 0 \quad \textbf{x} \in \partial \Omega,
# ```
# where $\partial \Omega$ denotes the boundary of $\Omega$.
# The resulting weak form is given given as follows: Find ``u \in \mathbb{U}`` such that
# ```math
# \int_{\Omega} \nabla \delta u \cdot \nabla u \ d\Omega = \int_{\Omega} \delta u \ d\Omega \quad \forall \delta u \in \mathbb{T},
# ```
# where $\delta u$ is a test function, and where $\mathbb{U}$ and $\mathbb{T}$ are suitable
# trial and test function sets, respectively.
#-
# ## Commented Program
#
# Now we solve the problem in Ferrite. What follows is a program spliced with comments.
#md # The full program, without comments, can be found in the next [section](@ref heat_equation-plain-program).
#
# First we load Ferrite, and some other packages we need
using Ferrite, CUDA
using IterativeSolvers, LinearAlgebra

### TODO Extension
import Adapt
using StaticArrays
struct GPUQuadratureRule{N,T,dim}
    weights::SVector{N,T}
    points::SVector{N,Vec{dim,T}}
end
function Adapt.adapt_structure(to, qr::QuadratureRule{shape,T,dim}) where {shape,T,dim}
    N = length(qr.weights)
    GPUQuadratureRule{N,T,dim}(SVector{N,T}(qr.weights), SVector{N,Vec{dim,T}}(qr.points))
end
function Adapt.adapt_structure(to, nodes::Vector{Node})
    CuArray(get_node_coordinate.(nodes))
end
### TODO Adapt dofhandler?

# TODO not sure how to do this automatically
function unsafe_shape_value(ip::Lagrange{RefQuadrilateral, 1}, ξ::Vec{2}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    i == 1 && return (1 - ξ_x) * (1 - ξ_y) / 4
    i == 2 && return (1 + ξ_x) * (1 - ξ_y) / 4
    i == 3 && return (1 + ξ_x) * (1 + ξ_y) / 4
    i == 4 && return (1 - ξ_x) * (1 + ξ_y) / 4
end

function unsafe_shape_value(ip::Lagrange{RefHexahedron, 1}, ξ::Vec{3}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    ξ_z = ξ[3]
    i == 1 && return 0.125(1 - ξ_x) * (1 - ξ_y) * (1 - ξ_z)
    i == 2 && return 0.125(1 + ξ_x) * (1 - ξ_y) * (1 - ξ_z)
    i == 3 && return 0.125(1 + ξ_x) * (1 + ξ_y) * (1 - ξ_z)
    i == 4 && return 0.125(1 - ξ_x) * (1 + ξ_y) * (1 - ξ_z)
    i == 5 && return 0.125(1 - ξ_x) * (1 - ξ_y) * (1 + ξ_z)
    i == 6 && return 0.125(1 + ξ_x) * (1 - ξ_y) * (1 + ξ_z)
    i == 7 && return 0.125(1 + ξ_x) * (1 + ξ_y) * (1 + ξ_z)
    i == 8 && return 0.125(1 - ξ_x) * (1 + ξ_y) * (1 + ξ_z)
end

function shape_values(ip::Lagrange{RefQuadrilateral, 1}, ξ::Vec{2})
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    return @SVector [
        (1 - ξ_x) * (1 - ξ_y) / 4,
        (1 + ξ_x) * (1 - ξ_y) / 4,
        (1 + ξ_x) * (1 + ξ_y) / 4,
        (1 - ξ_x) * (1 + ξ_y) / 4,
    ]
end

function shape_values(ip::Lagrange{RefHexahedron, 1}, ξ::Vec{3})
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    ξ_z = ξ[3]
    return @SVector [
        (1 - ξ_x) * (1 - ξ_y) * (1 - ξ_z) / 8,
        (1 + ξ_x) * (1 - ξ_y) * (1 - ξ_z) / 8,
        (1 + ξ_x) * (1 + ξ_y) * (1 - ξ_z) / 8,
        (1 - ξ_x) * (1 + ξ_y) * (1 - ξ_z) / 8,
        (1 - ξ_x) * (1 - ξ_y) * (1 + ξ_z) / 8,
        (1 + ξ_x) * (1 - ξ_y) * (1 + ξ_z) / 8,
        (1 + ξ_x) * (1 + ξ_y) * (1 + ξ_z) / 8,
        (1 - ξ_x) * (1 + ξ_y) * (1 + ξ_z) / 8,
    ]
end

function shape_gradients(ip::Lagrange{RefQuadrilateral, 1}, ξ::Vec{2})
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    return @SMatrix [
        (0 - 1) * (1 - ξ_y) / 4    (1 - ξ_x) * (0 - 1) / 4;
        (0 + 1) * (1 - ξ_y) / 4    (1 + ξ_x) * (0 - 1) / 4;
        (0 + 1) * (1 + ξ_y) / 4    (1 + ξ_x) * (0 + 1) / 4;
        (0 - 1) * (1 + ξ_y) / 4    (1 - ξ_x) * (0 + 1) / 4;
    ]
end

function shape_gradients(ip::Lagrange{RefHexahedron, 1}, ξ::Vec{3})
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    ξ_z = ξ[3]
    return @SMatrix [
        (0 - 1) * (1 - ξ_y) * (1 - ξ_z) / 8    (1 - ξ_x) * (0 - 1) * (1 - ξ_z) / 8    (1 - ξ_x) * (1 - ξ_y) * (0 - 1) / 8;
        (0 + 1) * (1 - ξ_y) * (1 - ξ_z) / 8    (1 + ξ_x) * (0 - 1) * (1 - ξ_z) / 8    (1 + ξ_x) * (1 - ξ_y) * (0 - 1) / 8;
        (0 + 1) * (1 + ξ_y) * (1 - ξ_z) / 8    (1 + ξ_x) * (0 + 1) * (1 - ξ_z) / 8    (1 + ξ_x) * (1 + ξ_y) * (0 - 1) / 8;
        (0 - 1) * (1 + ξ_y) * (1 - ξ_z) / 8    (1 - ξ_x) * (0 + 1) * (1 - ξ_z) / 8    (1 - ξ_x) * (1 + ξ_y) * (0 - 1) / 8;
        (0 - 1) * (1 - ξ_y) * (1 + ξ_z) / 8    (1 - ξ_x) * (0 - 1) * (1 + ξ_z) / 8    (1 - ξ_x) * (1 - ξ_y) * (0 + 1) / 8;
        (0 + 1) * (1 - ξ_y) * (1 + ξ_z) / 8    (1 + ξ_x) * (0 - 1) * (1 + ξ_z) / 8    (1 + ξ_x) * (1 - ξ_y) * (0 + 1) / 8;
        (0 + 1) * (1 + ξ_y) * (1 + ξ_z) / 8    (1 + ξ_x) * (0 + 1) * (1 + ξ_z) / 8    (1 + ξ_x) * (1 + ξ_y) * (0 + 1) / 8;
        (0 - 1) * (1 + ξ_y) * (1 + ξ_z) / 8    (1 - ξ_x) * (0 + 1) * (1 + ξ_z) / 8    (1 - ξ_x) * (1 + ξ_y) * (0 + 1) / 8;
    ]
end

function unsafe_shape_gradient(ip::Interpolation, ξ::Vec, i::Int)
    return Tensors.gradient(x -> unsafe_shape_value(ip, x, i), ξ)
end

# ntuple fails...
cellnodes(cell::Quadrilateral, nodes) = (nodes[cell.nodes[1]], nodes[cell.nodes[2]], nodes[cell.nodes[3]], nodes[cell.nodes[4]])
cellnodes(cell::Hexahedron, nodes) = (nodes[cell.nodes[1]], nodes[cell.nodes[2]], nodes[cell.nodes[3]], nodes[cell.nodes[4]], nodes[cell.nodes[5]], nodes[cell.nodes[6]], nodes[cell.nodes[7]], nodes[cell.nodes[8]])

# We start by generating a simple grid with 20x20 quadrilateral elements
# using `generate_grid`. The generator defaults to the unit square,
# so we don't need to specify the corners of the domain.
grid = generate_grid(Hexahedron, (100, 100, 100));
colors = CuArray.(create_coloring(grid)); # TODO add example without coloring, i.e. using Atomics instead

# ### Trial and test functions
# A `CellValues` facilitates the process of evaluating values and gradients of
# test and trial functions (among other things). To define
# this we need to specify an interpolation space for the shape functions.
# We use Lagrange functions
# based on the two-dimensional reference quadrilateral. We also define a quadrature rule based on
# the same reference element. We combine the interpolation and the quadrature rule
# to a `CellValues` object.
ip = Lagrange{RefHexahedron, 1}()
qr = QuadratureRule{RefHexahedron}(2)
cellvalues = CellValues(qr, ip);

# ### Degrees of freedom
# Next we need to define a `DofHandler`, which will take care of numbering
# and distribution of degrees of freedom for our approximated fields.
# We create the `DofHandler` and then add a single scalar field called `:u` based on
# our interpolation `ip` defined above.
# Lastly we `close!` the `DofHandler`, it is now that the dofs are distributed
# for all the elements.
dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh);

# ### Boundary conditions
# In Ferrite constraints like Dirichlet boundary conditions
# are handled by a `ConstraintHandler`.
# ch = ConstraintHandler(dh);

# Next we need to add constraints to `ch`. For this problem we define
# homogeneous Dirichlet boundary conditions on the whole boundary, i.e.
# the `union` of all the face sets on the boundary.
# ∂Ω = union(
#     getfaceset(grid, "left"),
#     getfaceset(grid, "right"),
#     getfaceset(grid, "top"),
#     getfaceset(grid, "bottom"),
# );

# Now we are set up to define our constraint. We specify which field
# the condition is for, and our combined face set `∂Ω`. The last
# argument is a function of the form $f(\textbf{x})$ or $f(\textbf{x}, t)$,
# where $\textbf{x}$ is the spatial coordinate and
# $t$ the current time, and returns the prescribed value. Since the boundary condition in
# this case do not depend on time we define our function as $f(\textbf{x}) = 0$, i.e.
# no matter what $\textbf{x}$ we return $0$. When we have
# specified our constraint we `add!` it to `ch`.
# dbc = Dirichlet(:u, ∂Ω, (x, t) -> 0)
# add!(ch, dbc);
# close!(ch)

function gpu_element_mass_action!(uₑout::V, uₑin::V, qr::GPUQuadratureRule{N,T,dim}, ip, ip_geo, xₑ) where {V,N,T,dim}
    n_basefuncs = length(uₑin) #getnbasefunctions(cellvalues)
    for q_point in 1:length(qr.weights) #getnquadpoints(cellvalues)
        ξ = qr.points[q_point]
        # TODO recover abstraction layer
        J = getJ(ip_geo, xₑ, ξ)
        dΩ = det(J)*qr.weights[q_point] #getdetJdV(cellvalues, q_point)
        for i in 1:n_basefuncs
            ϕᵢ = unsafe_shape_value(ip, ξ, i)
            for j in 1:n_basefuncs
                ϕⱼ = unsafe_shape_value(ip, ξ, j)
                uₑout[i] += ϕᵢ*ϕⱼ*uₑin[j]*dΩ
            end
        end
    end
    return nothing
end

function gpu_element_mass_action2!(uₑout::V, uₑin::V, qr::GPUQuadratureRule{N,T,dim}, ip::IP, ip_geo::GIP, xₑ) where {V,N,T,dim,IP,GIP}
    n_basefuncs = length(uₑin) #getnbasefunctions(cellvalues)
    for q_point in 1:length(qr.weights) #getnquadpoints(cellvalues)
        ξ = qr.points[q_point]
        shapes = shape_values(ip, ξ)
        # TODO recover abstraction layer
        J = getJ(ip_geo, xₑ, ξ)
        dΩ = det(J)*qr.weights[q_point] #getdetJdV(cellvalues, q_point)
        for i in 1:n_basefuncs
            ϕᵢ = shapes[i]
            for j in 1:n_basefuncs
                inner = shapes[j]*uₑin[j]
                uₑout[i] += inner*ϕᵢ*dΩ
            end
        end
    end
    return nothing
end

# Mass action on a single color
Base.@propagate_inbounds function gpu_mass_action_kernel!(uout::V, uin::V, all_cell_dofs, cell_dof_offsets, cell_indices, qr::GPUQuadratureRule{N,T,dim}, ip::IP, ip_geo::GIP, cells::CuDeviceArray{CT}, nodes::CuDeviceArray{Vec{dim,T}}, dofs_per_cell)  where {V,N,T,dim,CT,IP,GIP}
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    for i = index:stride:length(cell_indices)
        ## Get index of the current cell
        cell_index = cell_indices[i]
        ## Grab the actual cell
        cell = cells[cell_index]
        ## Grab the dofs on the cell
        cell_dof_range = cell_dof_offsets[cell_index]:(cell_dof_offsets[cell_index]+dofs_per_cell-1)
        cell_dofs = @view all_cell_dofs[cell_dof_range]
        ## Grab the buffers for the y and x
        uₑin  = @view uin[cell_dofs]
        uₑout = @view uout[cell_dofs]
        ## Grab coordinate array
        xₑ = cellnodes(cell, nodes)
        ## Apply local action for y = Ax
        gpu_element_mass_action2!(uₑout, uₑin, qr, ip, ip_geo, xₑ)
    end
    return nothing
end

# Mass action of the full operator
function gpu_mass_action!(uout::V, u::V, all_cell_dofs, cell_dof_offsets, qr::QR, ip::IP, ip_geo::GIP, colors, gpu_cells, gpu_nodes) where {V, QR, IP, GIP}
    # Initialize solution
    fill!(uout, 0.0)
    synchronize()

    # Apply action one time
    dofs_per_cell  = ndofs_per_cell(dh, 1)
    for color ∈ colors
        numthreads = 256
        numblocks = 1 #fails...? ceil(Int, length(color)/numthreads)
        # try
            @cuda threads=numthreads blocks=numblocks gpu_mass_action_kernel!(uout, u, all_cell_dofs, cell_dof_offsets, color, qr, ip, ip_geo, gpu_cells, gpu_nodes, dofs_per_cell)
            synchronize()
            # catch err
        #     code_typed(err; interactive = true)
        # end
    end
end

function getJ(ip_geo::GIP, xₑ::NTuple{N,Vec{dim,T}}, ξ::Vec{dim,T}) where {N,GIP,dim,T}
    dMdξ = shape_gradients(ip_geo, ξ)
    fecv_J = zero(Tensor{2,dim,T})
    for i in 1:length(xₑ)
        fecv_J += xₑ[i] ⊗ Vec{dim,T}(dMdξ[i, :])
    end
    return fecv_J
end

# 
# Define the operator struct (https://iterativesolvers.julialinearalgebra.org/dev/getting_started/)
struct FerriteGPUMassOperator{CACELLS, CANODES, CACOLORS, CADOFS, CAOFFSETS, IP, GIP, QR}
    # "GPUGrid"
    gpu_cells::CACELLS
    gpu_nodes::CANODES
    colors::CACOLORS
    # "GPUDofHandler"
    all_cell_dofs::CADOFS
    cell_dof_offsets::CAOFFSETS
    # "GPUValues"
    ip::IP
    ip_geo::GIP
    qr::QR
end

function FerriteGPUMassOperator(dh, colors, ip, ip_geo, qr)
    FerriteGPUMassOperator(
        CuArray(getcells(Ferrite.get_grid(dh))),
        CuArray(get_node_coordinate.(getnodes(Ferrite.get_grid(dh)))),
        colors,
        CuArray(dh.cell_dofs),
        CuArray(dh.cell_dofs_offset),
        ip, ip_geo, qr
    )
end

LinearAlgebra.mul!(y, A::FerriteGPUMassOperator, x) = gpu_mass_action!(y, x, A.all_cell_dofs, A.cell_dof_offsets, A.qr, A.ip, A.ip_geo, A.colors, A.gpu_cells, A.gpu_nodes)
Base.eltype(A::FerriteGPUMassOperator) = Float64
Base.size(A::FerriteGPUMassOperator, d) = ndofs(dh) # Square operator

A = FerriteGPUMassOperator(dh, colors, ip, ip, qr)

struct FerriteGPURHS{CACELLS, CANODES, CACOLORS, CADOFS, CAOFFSETS, IP, GIP, QR}
    # "GPUGrid"
    gpu_cells::CACELLS
    gpu_nodes::CANODES
    colors::CACOLORS
    # "GPUDofHandler"
    all_cell_dofs::CADOFS
    cell_dof_offsets::CAOFFSETS
    # "GPUValues"
    ip::IP
    ip_geo::GIP
    qr::QR
end

function FerriteGPURHS(dh, colors, ip, ip_geo, qr)
    FerriteGPURHS(
        CuArray(getcells(Ferrite.get_grid(dh))),
        CuArray(get_node_coordinate.(getnodes(Ferrite.get_grid(dh)))),
        colors,
        CuArray(dh.cell_dofs),
        CuArray(dh.cell_dofs_offset),
        ip, ip_geo, qr
    )
end

function generate_rhs(gpurhs::FerriteGPURHS)
    rhs = CuArray(zeros(ndofs(dh)))
    numthreads = 256
    numblocks = 1 #fails...? ceil(Int, length(color)/numthreads)
    dofs_per_cell  = ndofs_per_cell(dh, 1)
    for color ∈ gpurhs.colors
        # try
        @cuda threads=numthreads blocks=numblocks gpu_rhs_kernel!(rhs, gpurhs.all_cell_dofs, gpurhs.cell_dof_offsets, gpurhs.qr, gpurhs.ip, gpurhs.ip_geo, color, gpurhs.gpu_cells, gpurhs.gpu_nodes, dofs_per_cell)
        synchronize()
        # catch err
        #     code_typed(err; interactive = true)
        # end
    end

    # for cell in CellIterator(dh)
    #     reinit!(cellvalues, cell)
    #     coords = getcoordinates(cell)
    #     n_basefuncs = getnbasefunctions(cellvalues)
    #     fe = zeros(n_basefuncs)
    #     for q_point in 1:getnquadpoints(cellvalues)
    #         ## Get the quadrature weight
    #         dΩ = getdetJdV(cellvalues, q_point)
    #         x = spatial_coordinate(cellvalues, q_point, coords)
    #         ## Loop over test shape functions
    #         for i in 1:n_basefuncs
    #             δu  = shape_value(cellvalues, q_point, i)
    #             ## Add contribution to fe
    #             fe[i] += cos(x[1]/π)*cos(x[2]/π)*δu * dΩ
    #         end
    #     end
    #     rhs[celldofs(cell)] .+= fe
    # end
    return rhs
end

Base.@propagate_inbounds function gpu_rhs_kernel!(rhs::RHS, all_cell_dofs::CD, cell_dof_offsets::DO, qr::QR, ip::FIP, ip_geo::GIP, cell_indices::GPUC, cells::CELLS, nodes::GPUN, dofs_per_cell::DOFSPC) where {RHS, CD, DO, QR, FIP, GIP, GPUC, CELLS, GPUN, DOFSPC}
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    for i = index:stride:length(cell_indices)
        ## Get index of the current cell
        cell_index = cell_indices[i]
        ## Grab the actual cell
        cell = cells[cell_index]
        ## Grab the dofs on the cell
        cell_dof_range = cell_dof_offsets[cell_index]:(cell_dof_offsets[cell_index]+dofs_per_cell-1)
        cell_dofs = @view all_cell_dofs[cell_dof_range]
        ## Grab the buffers for the y and x
        rhsₑ = @view rhs[cell_dofs]
        ## Grab coordinate array
        coords = cellnodes(cell, nodes)
        ## Apply local action for y = Ax
        gpu_rhs_kernel2!(rhsₑ, qr, ip, ip_geo, coords)
    end
end

function gpu_rhs_kernel2!(rhsₑ, qr, ip, ip_geo, coords)    
    n_basefuncs = getnbasefunctions(ip)
    for q_point in 1:length(qr.weights) #getnquadpoints(cellvalues)
        ξ = qr.points[q_point]
        # TODO recover abstraction layer
        J = getJ(ip_geo, coords, ξ)
        dΩ = det(J)*qr.weights[q_point] #getdetJdV(cellvalues, q_point)
        
        # TODO spatial_coordinate
        x = zero(Vec{3,Float64})
        for i in 1:n_basefuncs
            x += unsafe_shape_value(ip_geo, ξ, i)*coords[i] #geometric_value(fe_v, q_point, i) * x[i]
        end

        for i in 1:n_basefuncs
            ϕᵢ = unsafe_shape_value(ip, ξ, i)
            rhsₑ[i] += cos(x[1]/π)*cos(x[2]/π)*ϕᵢ * dΩ
        end
    end
end


b = generate_rhs(FerriteGPURHS(dh, colors, ip, ip, qr))
u = CUDA.fill(0.0, ndofs(dh));
cg!(u, A, b; verbose=true);

# ### Exporting to VTK
# To visualize the result we export the grid and our field `u`
# to a VTK-file, which can be viewed in e.g. [ParaView](https://www.paraview.org/).
vtk_grid("heat_equation", dh) do vtk # The cake is a lie
    vtk_point_data(vtk, dh, Array(u))
end

## test the result                #src
# using Test                        #src
# @test norm(u) ≈ 3.307743912641305 #src

#md # ## [Plain program](@id heat_equation-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`heat_equation.jl`](heat_equation.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
