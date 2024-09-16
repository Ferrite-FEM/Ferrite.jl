#=
Implementation of the heat equation using the GPU using two kernels; the first one is to set the local stiffness matrix and force vector,
and the second one is to assemble the global stiffness matrix and force vector,where each component of the local stiffness matrix is
assembled in the global matrix by a thread.
=#

using Ferrite, CUDA
using KernelAbstractions
using StaticArrays
using SparseArrays
using Adapt
using Test
using NVTX



left = Tensor{1,2,Float32}((0,-0)) # define the left bottom corner of the grid.
right = Tensor{1,2,Float32}((10.0,10.0)) # define the right top corner of the grid.


grid = generate_grid(Quadrilateral, (10, 10),left,right)


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




@kernel function assemble_local_gpu(kes,fes,cv,dh)

    e = @index(Global) |> Int32
    n_basefuncs = getnbasefunctions(cv)
    # e is the global index of the finite element in the grid.
    cell_coords = getcoordinates(dh.grid, e)

    ke = @view kes[e,:,:]
    fe = @view fes[e,:]
    # Loop over quadrature points
     for qv in Ferrite.QuadratureValuesIterator(cv,cell_coords)
        ## Get the quadrature weight
        dΩ = getdetJdV(qv)
        ## Loop over test shape functions
        for j in 1:n_basefuncs
            δu  = shape_value(qv, j)
            ∇δu = shape_gradient(qv, j)
            ## Add contribution to fe
            fe[j] += δu * dΩ
            ## Loop over trial shape functions
            for i in 1:n_basefuncs
                ∇u = shape_gradient(qv, i)
                ## Add contribution to Ke
                ke[i,j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
end



@kernel function assemble_global_gpu!(assembler,kes,fes,dh)
    # e is element index
    # ty, tz are local indices of the local matrices
    e, ty, tz = @index(Global,NTuple) .|> Int32
    dofs = celldofs(dh, e)

    jg = dofs[ty]
    ig = dofs[tz]
    if tz == Int32(1)
        assemble_atomic!(assembler,kes[e,ty,tz],fes[e,ty],ig,jg)
    else
        assemble_atomic!(assembler,kes[e,ty,tz],ig,jg)
    end
end


function allocate_local_matrices(backend,n_cells,cv)
    n_basefuncs = getnbasefunctions(cv)
    ke = KernelAbstractions.zeros(backend,Float32,n_cells , n_basefuncs, n_basefuncs)
    fe = KernelAbstractions.zeros(backend,Float32,n_cells, n_basefuncs)
    return ke,fe
end


Adapt.@adapt_structure Ferrite.GPUGrid
Adapt.@adapt_structure Ferrite.GPUDofHandler
Adapt.@adapt_structure Ferrite.GPUAssemblerSparsityPattern


#=NVTX.@annotate=# function assemble_global_gpu(backend,cellvalues,dh)
    n_cells = dh |> get_grid |> getncells |> Int32
    kes,fes = allocate_local_matrices(backend,n_cells,cellvalues)
    K = allocate_matrix(SparseMatrixCSC{Float32, Int32},dh)
    Kgpu = CUSPARSE.CuSparseMatrixCSC(K)
    fgpu = KernelAbstractions.zeros(backend,Float32,ndofs(dh))
    assembler = start_assemble(Kgpu, fgpu)
    # set up kernel adaption & launch the kernel
    dh_gpu = Adapt.adapt_structure(CuArray, dh)
    assembler_gpu = Adapt.adapt_structure(CUDA.KernelAdaptor(), assembler)
    cellvalues_gpu = Adapt.adapt_structure(CuArray, cellvalues)
    # assemble the local matrices in kes and fes
    kernel_local = assemble_local_gpu(backend)
    kernel_local(kes,fes,cellvalues,dh_gpu;  ndrange =n_cells)
    # assemble the global matrix
    n_basefuncs = getnbasefunctions(cellvalues)
    kernel_global = assemble_global_gpu!(backend)

    kernel_global(assembler_gpu,kes,fes,dh_gpu;  ndrange = (n_cells,n_basefuncs,n_basefuncs))

    return Kgpu,fgpu
end


stassy(cv,dh) = assemble_global!(cv,dh,Val(false))




# qpassy(cv,dh) = assemble_global!(cv,dh,Val(true))

backend  = CUDABackend();
Kgpu, fgpu =assemble_global_gpu(backend,cellvalues,dh);

#using BenchmarkTools

#Kgpu, fgpu = @btime CUDA.@sync    assemble_global_gpu($cellvalues,$dh);
#Kstd , Fstd =@btime  stassy($cellvalues,$dh);
# to benchmark the code using nsight compute use the following command: ncu --mode=launch julia
# Open nsight compute and attach the profiler to the julia instance
# ref: https://cuda.juliagpu.org/v2.2/development/profiling/#NVIDIA-Nsight-Compute
# to benchmark using nsight system use the following command: # nsys profile --trace=nvtx julia rmse_kernel_v1.jl


norm(Kgpu)


Kstd , Fstd =stassy(cellvalues,dh);
norm(Kstd)

@testset "GPU Heat Equation" begin

    for i = 1:10
        # Bottom left point in the grid in the physical coordinate system.
        # Generate random Float32 between -100 and -1
        bl_x = rand(Float32) * (-99) - 1
        bl_y = rand(Float32) * (-99) - 1

        # Top right point in the grid in the physical coordinate system.
        # Generate random Float32 between 0 and 100
        tr_x = rand(Float32) * 100
        tr_y = rand(Float32) * 100

        n_x = rand(1:100)   # number of cells in x direction
        n_y = rand(1:100)   # number of cells in y direction

        left = Tensor{1,2,Float32}((bl_x,bl_y)) # define the left bottom corner of the grid.
        right = Tensor{1,2,Float32}((tr_x,tr_y)) # define the right top corner of the grid.


        grid = generate_grid(Quadrilateral, (n_x, n_y),left,right)


        colors = create_coloring(grid) .|> (x -> Int32.(x)) # convert to Int32 to reduce number of registers


        ip = Lagrange{RefQuadrilateral, 1}() # define the interpolation function (i.e. Bilinear lagrange)


        qr = QuadratureRule{RefQuadrilateral,Float32}(2)


        cellvalues = CellValues(Float32,qr, ip)


        dh = DofHandler(grid)



        add!(dh, :u, ip)

        close!(dh);
        # The CPU version:
        Kstd , Fstd =  stassy(cellvalues,dh);

        # The GPU version
        Kgpu, fgpu =  assemble_global_gpu(cellvalues,dh,colors)

        @test norm(Kstd) ≈ norm(Kgpu) atol=1e-4
    end
end
