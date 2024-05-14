using Ferrite, CUDA
using StaticArrays
using Adapt

left = Tensor{1,2,Float64}((0,-0)) # define the left bottom corner of the grid.
right = Tensor{1,2,Float64}((400.0,400.0)) # define the right top corner of the grid.
grid = generate_grid(Quadrilateral, (100, 100),left,right); 



ip = Lagrange{RefQuadrilateral, 1}() # define the interpolation function (i.e. Bilinear lagrange)

# define the numerical integration rule 
# (i.e. integrating over quad shape with two quadrature points per direction)
qr = QuadratureRule{RefQuadrilateral}(2) 
cellvalues = CellValues(qr, ip);
static_cellvalues = Ferrite.StaticCellValues(cellvalues, Val(true));
size(static_cellvalues.fv.Nξ,1)
# Notes about cell values regarding gpu:
# 1. fun_values & geo_mapping in CellValues are not bits object. Therefore, they cannot be put on the gpu.
# 2. fv & gm in StaticCellValues are bits object. Therefore, they can be put on the gpu.
# 3. StaticCellValues can be a bitstype be reomoving x property from it. 


dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh);



# Standard assembly of the element.
function assemble_element_std!(Ke::Matrix, fe::Vector, cellvalues::CellValues)
    n_basefuncs = getnbasefunctions(cellvalues)

    # Loop over quadrature points
    for q_point in 1:getnquadpoints(cellvalues)
        # Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        # Loop over test shape functions
        for i in 1:n_basefuncs
            δu  = shape_value(cellvalues, q_point, i)
            ∇δu = shape_gradient(cellvalues, q_point, i)
            # Add contribution to fe
            fe[i] += δu * dΩ
            # Loop over trial shape functions
            for j in 1:n_basefuncs
                ∇u = shape_gradient(cellvalues, q_point, j)
                # Add contribution to Ke
                Ke[i, j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
    return Ke, fe
end


# Element assembly by using static cell (PR #883)
function assemble_element_qpiter!(Ke::Matrix, fe::Vector, cellvalues)
    n_basefuncs = getnbasefunctions(cellvalues)
    ## Loop over quadrature points
    for qv in Ferrite.QuadratureValuesIterator(cellvalues)
        ## Get the quadrature weight
         dΩ = getdetJdV(qv)
        ## Loop over test shape functions
        for i in 1:n_basefuncs
            δu  = shape_value(qv, i)
            ∇δu = shape_gradient(qv, i)
            ## Add contribution to fe
            fe[i] += δu * dΩ
            ## Loop over trial shape functions
            for j in 1:n_basefuncs
                ∇u = shape_gradient(qv, j)
                ## Add contribution to Ke
                Ke[i, j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
    return Ke, fe
end


function create_buffers(cellvalues, dh)
    f = zeros(ndofs(dh))
    K = create_sparsity_pattern(dh)
    assembler = start_assemble(K, f)
    ## Local quantities
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    return (;f, K, assembler, Ke, fe)
end


# Standard global assembly
function assemble_global!(cellvalues, dh::DofHandler,qp_iter::Val{QPiter}) where {QPiter}
    (;f, K, assembler, Ke, fe) = create_buffers(cellvalues,dh)
    # Loop over all cels
    for cell in CellIterator(dh)
        fill!(Ke, 0)
        fill!(fe, 0)
        if QPiter
            reinit!(cellvalues, getcoordinates(cell))
            assemble_element_qpiter!(Ke, fe, cellvalues)
        else
            # Reinitialize cellvalues for this cell
            reinit!(cellvalues, cell)
            # Compute element contribution
            assemble_element_std!(Ke, fe, cellvalues)
        end
        # Assemble Ke and fe into K and f
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    return K, f
end



# Helper function to get all the coordinates from the grid.
function get_all_coordinates(grid::Ferrite.AbstractGrid{dim}) where {dim}
    coords = Vector{Vec{2,Float32}}() 
    n_cells = length(grid.cells)
    for i = 1:n_cells
        append!(coords,getcoordinates(grid,i))
    end
    coords
end

struct GPUGrid{sdim,V<:Vec{sdim,Float32},COORDS<:AbstractArray{V,1}} <: Ferrite.AbstractGrid{sdim} 
    all_coords::COORDS
    n_cells::Int32
    
end

function GPUGrid(grid::Grid{sdim}) where sdim
    all_coords = cu(get_all_coordinates(grid))
    n_cells = Int32(length(grid.cells))
    GPUGrid(all_coords,n_cells)
end


struct GPUDofHandler{CDOFS<:AbstractArray{<:Number,1},GRID<:GPUGrid}<: Ferrite.AbstractDofHandler
    cell_dofs::CDOFS
    grid::GRID
end


function GPUDofHandler(dh::DofHandler)
    GPUDofHandler(cu(Int32.(dh.cell_dofs)),GPUGrid(dh.grid))
end

function Adapt.adapt_structure(to, grid::GPUGrid)
    all_coords = Adapt.adapt_structure(to, grid.all_coords)
    n_cells = Adapt.adapt_structure(to, grid.n_cells)
    GPUGrid(all_coords, n_cells)
end

function Adapt.adapt_structure(to, dh::GPUDofHandler)
    cell_dofs = Adapt.adapt_structure(to, dh.cell_dofs)
    grid = Adapt.adapt_structure(to, dh.grid)
    GPUDofHandler(cell_dofs, grid)
end

function Adapt.adapt_structure(to, cv::Ferrite.StaticCellValues)
    fv = Adapt.adapt_structure(to, cv.fv)
    gm = Adapt.adapt_structure(to, cv.gm)
    x = Adapt.adapt_structure(to, cu(cv.x))
    weights = Adapt.adapt_structure(to, cv.weights)
    Ferrite.StaticCellValues(fv,gm,x, weights)
end


gm = static_cellvalues.gm
x = get_all_coordinates(grid)
J = gm.dNdξ[1,1] ⊗ x[1] + gm.dNdξ[2,1] ⊗ x[2] + gm.dNdξ[3,1] ⊗ x[3] + gm.dNdξ[4,1] ⊗ x[4]
det(J)
inv_J = inv(J)
inv_J ⋅ static_cellvalues.fv.dNdξ[1,1]

function getjacobian(gm::Ferrite.StaticInterpolationValues,x,qr)
    n_basefuncs = size(gm.Nξ,1)
    J = gm.dNdξ[1,qr] ⊗ x[1]
    for i = 2:n_basefuncs 
        J+= gm.dNdξ[i,qr] ⊗ x[i]
    end
    return J
end

function assemble_element_gpu!(Kgpu,cv::Ferrite.StaticCellValues,dh::GPUDofHandler) 
    tx = threadIdx().x 
    bx = blockIdx().x
    bd = blockDim().x
    e = tx + (bx-1)*bd
    n_cells = dh.grid.n_cells
    e ≤ n_cells || return nothing # e here is the current element index.
    n_qr = length(cv.weights)
    n_basefuncs = size(cv.fv.Nξ,1)
    dofs = dh.cell_dofs
    x = dh.grid.all_coords
    for qr = 1:n_qr # loop over quadrature points # TODO: propogate_ibounds
        si = (e-1)*n_basefuncs
        #J = gm.dNdξ[1,qr] ⊗ x[si+1] + gm.dNdξ[2,qr] ⊗ x[si+2] + gm.dNdξ[3,qr] ⊗ x[si+3] + gm.dNdξ[4,qr] ⊗ x[si+4]
        cell_x = @view x[si+1:si+size(cv.gm.Nξ,1)]
        J = getjacobian(cv.gm, cell_x,qr)
        inv_J = inv(J)
        #@cushow det(J)
        @inbounds dΩ = det(J) * cv.weights[qr] 
        for i = 1:n_basefuncs
          @inbounds  ∇δu = inv_J ⋅ cv.fv.dNdξ[i,qr]
            for j = 1:n_basefuncs
               @inbounds ∇u = inv_J ⋅ cv.fv.dNdξ[j,qr]
               @inbounds ig = dofs[(e-1)*n_basefuncs+i]
               @inbounds jg = dofs[(e-1)*n_basefuncs+j]
               CUDA.@atomic  Kgpu[ig, jg] += (∇δu ⋅ ∇u) * dΩ # atomic because many threads might write into the same memory addrres at the same time. 
            end
        end
    end
    return nothing
end

Kgpu = CUDA.zeros(dh.ndofs.x,dh.ndofs.x)
gpu_dh = GPUDofHandler(dh)



function assemble_global_gpu!(Kgpu)
    kernel = @cuda launch=false assemble_element_gpu!(Kgpu,static_cellvalues,gpu_dh)
    config = launch_configuration(kernel.fun)
    threads = min(length(grid.cells), config.threads)
    blocks =  cld(length(grid.cells), threads)
    kernel(Kgpu,static_cellvalues,gpu_dh; threads=threads, blocks=blocks)
end

stassy(cv,dh) = assemble_global!(cv,dh,Val(false))

qpassy(cv,dh) = assemble_global!(cv,dh,Val(true))

using BenchmarkTools
using LinearAlgebra


 assemble_global_gpu!(Kgpu)

 Kgpu
norm(Kgpu)

Kstd , Fstd =  stassy(cellvalues,dh);

norm(Kstd)

cvs = Ferrite.StaticCellValues(cellvalues, Val(true)) 

Kqp , Fqp =  qpassy(cvs,dh);

norm(Kqp)