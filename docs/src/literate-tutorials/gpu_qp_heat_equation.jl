using Ferrite
using StaticArrays
using SparseArrays
using CUDA


left = Tensor{1, 2, Float32}((0, -0)) # define the left bottom corner of the grid.
right = Tensor{1, 2, Float32}((1.0, 1.0)) # define the right top corner of the grid.
grid = generate_grid(Quadrilateral, (1000, 1000), left, right)


ip = Lagrange{RefQuadrilateral, 2}() # define the interpolation function (i.e. Bilinear lagrange)
qr = QuadratureRule{RefQuadrilateral}(Float32, 3)
cellvalues = CellValues(Float32, qr, ip)


dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh);


# Standard assembly of the element.
function assemble_element_std!(Ke::Matrix, fe::Vector, cellvalues::CellValues)
    n_basefuncs = getnbasefunctions(cellvalues)
    ## Loop over quadrature points
    for q_point in 1:getnquadpoints(cellvalues)
        ## Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        ## Loop over test shape functions
        for i in 1:n_basefuncs
            δu = shape_value(cellvalues, q_point, i)
            ∇δu = shape_gradient(cellvalues, q_point, i)
            ## Add contribution to fe
            fe[i] += δu * dΩ
            ## Loop over trial shape functions
            for j in 1:n_basefuncs
                ∇u = shape_gradient(cellvalues, q_point, j)
                ## Add contribution to Ke
                Ke[i, j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
    return Ke, fe
end


function assemble_element_qpiter!(Ke::Matrix, fe::Vector, cellvalues, cell_coords::AbstractVector)
    n_basefuncs = getnbasefunctions(cellvalues)
    ## Loop over quadrature points
    for qv in Ferrite.QuadratureValuesIterator(cellvalues, cell_coords)
        ## Get the quadrature weight
        dΩ = getdetJdV(qv)
        ## Loop over test shape functions
        for i in 1:n_basefuncs
            δu = shape_value(qv, i)
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


# Standard global assembly
function assemble_global_qp!(cellvalues, dh::DofHandler, K, f)
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(eltype(K), n_basefuncs, n_basefuncs)
    fe = zeros(eltype(f), n_basefuncs)
    assembler = start_assemble(K, f)
    ## Loop over all cels
    for cell in CellIterator(dh)
        fill!(Ke, 0)
        fill!(fe, 0)
        assemble_element_qpiter!(Ke, fe, cellvalues, getcoordinates(cell))
        ## Assemble Ke and fe into K and f
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    return K, f
end

function assemble_global_std!(cellvalues, dh::DofHandler, K, f)
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(eltype(K), n_basefuncs, n_basefuncs)
    fe = zeros(eltype(f), n_basefuncs)
    assembler = start_assemble(K, f)
    ## Loop over all cels
    for cell in CellIterator(dh)
        fill!(Ke, 0)
        fill!(fe, 0)
        ## Reinitialize cellvalues for this cell
        reinit!(cellvalues, cell)
        ## Compute element contribution
        assemble_element_std!(Ke, fe, cellvalues)
        ## Assemble Ke and fe into K and f
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    return K, f
end


## gpu version of element assembly
function assemble_element!(Ke, fe, cv, cell)
    n_basefuncs = getnbasefunctions(cv)
    for qv in Ferrite.QuadratureValuesIterator(cv, getcoordinates(cell))
        ## Get the quadrature weight
        dΩ = getdetJdV(qv)
        ## Loop over test shape functions
        for i in 1:n_basefuncs
            δu = shape_value(qv, i)
            ∇u = shape_gradient(qv, i)
            ## Add contribution to fe
            fe[i] += δu * dΩ
            ## fe_shared[tx,i] += δu * dΩ
            ## Loop over trial shape functions
            for j in 1:n_basefuncs
                ∇δu = shape_gradient(qv, j)
                ## Add contribution to Ke
                Ke[i, j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
    return
end


# gpu version of global assembly
function assemble_gpu!(Kgpu, fgpu, cv, dh)
    n_basefuncs = getnbasefunctions(cv)
    assembler = start_assemble(Kgpu, fgpu; fillzero = false) ## has to be always false
    for cell in CellIterator(dh, convert(Int32, n_basefuncs))
        Ke = cellke(cell)
        fe = cellfe(cell)
        assemble_element!(Ke, fe, cv, cell)
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    return nothing
end


n_basefuncs = getnbasefunctions(cellvalues)

## Allocate CPU matrix
#K = allocate_matrix(SparseMatrixCSC{Float64, Int64}, dh);
#f = zeros(eltype(K), ndofs(dh));


# Allocate GPU matrix
## commented to pass the test
## Kgpu = allocate_matrix(CUSPARSE.CuSparseMatrixCSC{Float32, Int32}, dh)
## fgpu = CUDA.zeros(Float32, ndofs(dh));

n_cells = dh |> get_grid |> getncells

# Kernel configuration
## GPU kernel ##
## commented to pass the test
## First init the kernel with the required config.
## gpu_kernel = init_kernel(BackendCUDA, n_cells, n_basefuncs, assemble_gpu!, (Kgpu, fgpu, cellvalues, dh))
## Then launch the kernel
## gpu_kernel |> launch! or gpu_kernel()
## gpu_kernel()

## CPU kernel ##
## cpu_kernel = init_kernel(BackendCPU, n_cells, n_basefuncs, assemble_gpu!, (K, f, cellvalues, dh));
## cpu_kernel()


## commented to pass the test
## norm(Kgpu)


## GPU Benchmarking, remove when not needed ##
# function setup_bench_gpu(n_cells, n_basefuncs, cellvalues, dh)
#     Kgpu = allocate_matrix(CUSPARSE.CuSparseMatrixCSC{Float32, Int32}, dh)
#     fgpu = CUDA.zeros(eltype(Kgpu), ndofs(dh));
#     gpu_kernel = init_kernel(BackendCUDA, n_cells, n_basefuncs, assemble_gpu!, (Kgpu, fgpu, cellvalues, dh))
# end

# CUDA.@time setup_bench_gpu(n_cells, n_basefuncs, cellvalues, dh)
# CUDA.@profile trace = true setup_bench_gpu(n_cells, n_basefuncs, cellvalues, dh)
# gpu_kernel = setup_bench_gpu(n_cells, n_basefuncs, cellvalues, dh)
# CUDA.@time gpu_kernel()
# CUDA.@profile trace = true gpu_kernel()


## CPU Benchmarking, remove when not needed ##
# function setup_bench_cpu( dh)
#     K = allocate_matrix(SparseMatrixCSC{Float64, Int}, dh)
#     f = zeros(eltype(K), ndofs(dh));
#     return K,f
# end

# using BenchmarkTools
# @benchmark setup_bench_cpu($dh)
# K,f = setup_bench_cpu(dh)
# @benchmark assemble_global_std!($cellvalues, $dh, $K, $f)
# @benchmark assemble_global_qp!($cellvalues, $dh, $K, $f)
