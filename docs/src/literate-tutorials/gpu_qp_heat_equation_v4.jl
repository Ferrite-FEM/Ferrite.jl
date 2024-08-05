#=
Implementationof the heat equation using the GPU using two kernels; the first one is to set the local stiffness matrix and force vector,
and the second one is to assemble the global stiffness matrix and force vector,where each component of the local stiffness matrix is
assembled in the global matrix by a thread.
=#

using Ferrite, CUDA
using StaticArrays
using SparseArrays
using Adapt
using Test
using NVTX






struct DofToElements{Ti, VEC_INT<:AbstractVector{Ti}}
    dof:: Ti
    elements:: VEC_INT # elements contain this global dof
    local_dofs::VEC_INT # local dofs of the global dof in each element
    n_elements::Ti
end


function Adapt.adapt_structure(to, dh::DofToElements)
    dof = Adapt.adapt_structure(to, dh.dof)
    elements = Adapt.adapt_structure(to, dh.elements |> cu)
    local_dofs = Adapt.adapt_structure(to, dh.local_dofs |> cu)
    n_elements = Adapt.adapt_structure(to, dh.n_elements)
    DofToElements(dof, elements, local_dofs, n_elements)
end


function map_dof_to_elements(dh::DofHandler, dof::Int)
    elements = []
    local_dofs = []
    ncells = dh |> get_grid |> getncells |> Int32
    for cell in 1:ncells
        dofs = celldofs(dh,cell)
        if dof ∈ dofs
            push!(elements, cell |> Int32)
            index = findfirst(e->e == dof, dofs) |> Int32
            push!(local_dofs,index)
        end
    end

    return DofToElements{Int32,Vector{Int32}}(Int32(dof), elements, local_dofs,length(elements) |> Int32)
end


function map_dofs_to_elements(dh::DofHandler)
    dofs = ndofs(dh)
    dofs_to_elements = range(1,dofs) .|> (dof -> map_dof_to_elements(dh, dof))
    return dofs_to_elements
end



left = Tensor{1,2,Float32}((0,-0)) # define the left bottom corner of the grid.
right = Tensor{1,2,Float32}((100.0,100.0)) # define the right top corner of the grid.


grid = generate_grid(Quadrilateral, (100, 100),left,right)


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


#=NVTX.@annotate=# function assemble_local_gpu(kes,fes,cv,dh,n_cells)
    tx = threadIdx().x
    bx = blockIdx().x
    bd = blockDim().x
    # e is the global index of the finite element in the grid.
    e = tx + (bx-Int32(1))*bd

    e ≤ n_cells || return nothing
    n_basefuncs = getnbasefunctions(cv)
    # e is the global index of the finite element in the grid.
    cell_coords = getcoordinates(dh.grid, e)

    ke = @view kes[e,:,:]
    fe = @view fes[e,:]
     #Loop over quadrature points
     for qv in Ferrite.QuadratureValuesIterator(cv,cell_coords)
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
                ke[i,j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
    return nothing
end


function assemble_global_gpu!(assembler,kes,fes,ndofs,dofs_to_elements)
    tx = threadIdx().x
    ty = threadIdx().y
    bx = blockIdx().x
    bd = blockDim().x

    dof_x = tx + (bx-Int32(1))*bd
    dof_y = ty + (bx-Int32(1))*bd
    dof_x ≤ ndofs || return nothing
    dof_y ≤ ndofs || return nothing
    k_val = 0.0f0
    f_val = 0.0f0
    dof_x_map = dofs_to_elements[dof_x]
    dof_y_map = dofs_to_elements[dof_y]
    nx = dof_x_map.n_elements
    ny = dof_y_map.n_elements
    for i in 1:nx
        e_x = dof_x_map.elements[i]
        for j in 1:ny
            e_y = dof_y_map.elements[j]
            if e_x == e_y
                local_dof_x = dof_x_map.local_dofs[i]
                local_dof_y = dof_y_map.local_dofs[j]
                k_val += kes[e_x,local_dof_x,local_dof_y]
                #f_val += fes[e_x,local_dof_x]
            end
        end
    end
    if ty == 1
        assemble!( assembler, k_val, f_val, dof_x, dof_y)
    else
        assemble!( assembler, k_val, dof_x, dof_y)
    end
    return nothing
end


function allocate_local_matrices(n_cells,cv)
    n_basefuncs = getnbasefunctions(cv)
    ke = CUDA.zeros(Float32,n_cells, n_basefuncs, n_basefuncs)
    fe = CUDA.zeros(Float32,n_cells, n_basefuncs)
    return ke,fe
end


Adapt.@adapt_structure Ferrite.GPUGrid
Adapt.@adapt_structure Ferrite.GPUDofHandler
Adapt.@adapt_structure Ferrite.GPUAssemblerSparsityPattern

#=NVTX.@annotate=# function assemble_global_gpu(cellvalues,dh)
    n_cells = dh |> get_grid |> getncells |> Int32
    kes,fes = allocate_local_matrices(n_cells,cellvalues)
    K = allocate_matrix(SparseMatrixCSC{Float32, Int32},dh)
    Kgpu = CUSPARSE.CuSparseMatrixCSC(K)
    fgpu = CUDA.zeros(ndofs(dh))
    assembler = start_assemble(Kgpu, fgpu)
    # set up kernel adaption & launch the kernel
    dh_gpu = Adapt.adapt_structure(CuArray, dh)
    assembler_gpu = Adapt.adapt_structure(CUDA.KernelAdaptor(), assembler)
    cellvalues_gpu = Adapt.adapt_structure(CuArray, cellvalues)
    # assemble the local matrices in kes and fes
    kernel_local = @cuda launch=false assemble_local_gpu(kes,fes,cellvalues_gpu,dh_gpu,n_cells)
    #@show CUDA.registers(kernel)
    config = launch_configuration(kernel_local.fun)
    threads = min(n_cells, config.threads)
    blocks =  cld(n_cells, threads)
    kernel_local(kes,fes,cellvalues,dh_gpu,n_cells;  threads, blocks)

    dofs_to_elements = map_dofs_to_elements(dh)
    # assemble the global matrix
    GC.@preserve  dofs_to_elements begin

        #dofs_to_elements = map_dofs_to_elements(dh) |> cu # map dofs to elements (element index, local dof index)
        dofs_to_elements = CuArray(cudaconvert.(dofs_to_elements))
        n_dofs = ndofs(dh)
        kernel_global = @cuda launch=false assemble_global_gpu!(assembler_gpu,kes,fes,n_dofs,dofs_to_elements)
        #CUDA.@device_code_warntype @cuda launch=false assemble_global_gpu!(assembler_gpu,kes,fes,n_dofs,dofs_to_elements)
        #@show CUDA.registers(kernel)
        config = launch_configuration(kernel_local.fun)
        threads_dofs = min(n_dofs, config.threads |> sqrt |> floor |> Int32)
        blocks =  cld(n_dofs, threads_dofs)
        kernel_global(assembler_gpu,kes,fes,n_dofs,dofs_to_elements;  threads = (threads_dofs,threads_dofs), blocks)

        return Kgpu,fgpu
    end
end


stassy(cv,dh) = assemble_global!(cv,dh,Val(false))




# qpassy(cv,dh) = assemble_global!(cv,dh,Val(true))

Kgpu, fgpu =assemble_global_gpu(cellvalues,dh);
#Kgpu, fgpu = CUDA.@profile    assemble_global_gpu_color(cellvalues,dh,colors)
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
