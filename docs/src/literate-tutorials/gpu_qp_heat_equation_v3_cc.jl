
using Ferrite, CUDA
using StaticArrays
using SparseArrays
using Adapt
using Test
using NVTX



left = Tensor{1,2,Float32}((0,-0)) # define the left bottom corner of the grid.
right = Tensor{1,2,Float32}((4.0,4.0)) # define the right top corner of the grid.


grid = generate_grid(Quadrilateral, (4, 4),left,right)


ip = Lagrange{RefQuadrilateral, 1}() # define the interpolation function (i.e. Bilinear lagrange)

qr = QuadratureRule{RefQuadrilateral}(Float32,2)


cellvalues = CellValues(Float32,qr, ip)


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


function create_buffers(cellvalues, dh)
    f = zeros(ndofs(dh))
    K = allocate_matrix(dh)
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
            #reinit!(cellvalues, getcoordinates(cell))
            assemble_element_qpiter!(Ke, fe, cellvalues,getcoordinates(cell))
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




#=NVTX.@annotate=# function assemble_gpu!(assembler,cv,dh,n_cells)
    tx = threadIdx().x
    bx = blockIdx().x
    bd = blockDim().x
    # e is the global index of the finite element in the grid.
    n_basefuncs = getnbasefunctions(cv)
    ke_shared = @cuDynamicSharedMem(Float32,(bd,n_basefuncs,n_basefuncs))
    fe_shared = @cuDynamicSharedMem(Float32,(bd,n_basefuncs),sizeof(Float32)*bd*n_basefuncs*n_basefuncs)
    fill!(ke_shared,0.0f0)
    fill!(fe_shared,0.0f0)
    sync_threads()

    e = tx + (bx-Int32(1))*bd

    e ≤ n_cells || return nothing
    # e is the global index of the finite element in the grid.
    # cell_coords = getcoordinates(dh.grid, e)
    cell = makecache(dh,e)
     #Loop over quadrature points
     for qv in Ferrite.QuadratureValuesIterator(cv,getcoordinates(cell))
        ## Get the quadrature weight
        dΩ = getdetJdV(qv)
        ## Loop over test shape functions
        for j in 1:n_basefuncs
            δu  = shape_value(qv, j)
            ∇u = shape_gradient(qv, j)
            ## Add contribution to fe
            #fe[j] += δu * dΩ
            fe_shared[tx,j] += δu * dΩ
            ## Loop over trial shape functions
            for i in 1:n_basefuncs
                ∇δu = shape_gradient(qv, i)
                ## Add contribution to Ke
                ke_shared[tx,i,j] += (∇δu ⋅ ∇u) * dΩ
                #ke[i,j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end


    dofs = celldofs(cell)
    for j in 1:n_basefuncs
        jg = dofs[j]
        assemble_atomic!(assembler,fe_shared[tx,j],jg)
        for i in 1:n_basefuncs
            ig = dofs[i]
            assemble_atomic!(assembler,ke_shared[tx,i,j],ig,jg)
        end
    end
    return nothing
end



Adapt.@adapt_structure Ferrite.GPUGrid
Adapt.@adapt_structure Ferrite.GPUDofHandler
Adapt.@adapt_structure Ferrite.GPUAssemblerSparsityPattern


function optimize_threads_for_dynshmem(max_threads, n_basefuncs)
    MAX_DYN_SHMEM = 48 * 1024 # TODO: get the maximum shared memory per block from the device (48KB for now, currently I don't know how to get this value)
    shmem_needed = sizeof(Float32) * (max_threads) * ( n_basefuncs) * n_basefuncs + sizeof(Float32) * (max_threads) * n_basefuncs
    if(shmem_needed < MAX_DYN_SHMEM)
        return max_threads, shmem_needed
    else
        # solve for threads
        max_possible = Int32(MAX_DYN_SHMEM ÷ (sizeof(Float32) * ( n_basefuncs) * n_basefuncs + sizeof(Float32) * n_basefuncs))
        dev = device()
        warp_size = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_WARP_SIZE)
        # approximate the number of threads to be a multiple of warp size (mostly 32)
        nearest_no_warps = max_possible ÷ warp_size
        if(nearest_no_warps < 4)
            throw(ArgumentError("Bad implementation (less than 4 warps per block, wasted resources)"))
        else
            possiple_threads = nearest_no_warps * warp_size
            shmem_needed = sizeof(Float32) * (possiple_threads) * ( n_basefuncs) * n_basefuncs + sizeof(Float32) * (possiple_threads) * n_basefuncs
            return possiple_threads, shmem_needed
        end
    end
end

#=NVTX.@annotate=# function assemble_gpu(cellvalues,dh)
    n_basefuncs = getnbasefunctions(cellvalues)
    n_cells = dh |> get_grid |> getncells |> Int32
    #kes,fes = allocate_local_matrices(n_cells,cellvalues)
    K = allocate_matrix(SparseMatrixCSC{Float32, Int32},dh)
    Kgpu = CUSPARSE.CuSparseMatrixCSC(K)
    fgpu = CUDA.zeros(ndofs(dh))
    assembler = start_assemble(Kgpu, fgpu)
    # set up kernel adaption & launch the kernel
    dh_gpu = Adapt.adapt_structure(CuArray, dh)
    assembler_gpu = Adapt.adapt_structure(CUDA.KernelAdaptor(), assembler)
    cellvalues_gpu = Adapt.adapt_structure(CuArray, cellvalues)
    # assemble the local matrices in kes and fes
    kernel = @cuda launch=false assemble_gpu!(assembler_gpu,cellvalues_gpu,dh_gpu,n_cells)
    #@show CUDA.registers(kernel)
    config = launch_configuration(kernel.fun)
    max_threads = min(n_cells, config.threads)
    threads, shared_mem = optimize_threads_for_dynshmem(max_threads, n_basefuncs)
    blocks =  cld(n_cells, threads)
    kernel(assembler_gpu,cellvalues,dh_gpu,n_cells;  threads, blocks, shmem=shared_mem)
    return Kgpu,fgpu
end


stassy(cv,dh) = assemble_global!(cv,dh,Val(false))




# qpassy(cv,dh) = assemble_global!(cv,dh,Val(true))

Kgpu, fgpu =assemble_gpu(cellvalues,dh);
#using BenchmarkTools

#Kgpu, fgpu = @btime CUDA.@sync    assemble_global_gpu($cellvalues,$dh);
#Kstd , Fstd =@btime  stassy($cellvalues,$dh);
#Kgpu, fgpu = CUDA.@profile  assemble_gpu(cellvalues,dh);
# to benchmark the code using nsight compute use the following command: ncu --mode=launch julia
# Open nsight compute and attach the profiler to the julia instance
# ref: https://cuda.juliagpu.org/v2.2/development/profiling/#NVIDIA-Nsight-Compute
# to benchmark using nsight system use the following command: # nsys profile --trace=nvtx julia rmse_kernel_v1.jl


norm(Kgpu)

Kstd , Fstd =stassy(cellvalues,dh);
norm(Kstd)
