using Ferrite, CUDA
using StaticArrays
using SparseArrays
using Adapt
using Test
using NVTX



left = Tensor{1,2,Float32}((0,-0)) # define the left bottom corner of the grid.
right = Tensor{1,2,Float32}((100.0,100.0)) # define the right top corner of the grid.


grid = generate_grid(Quadrilateral, (100, 100),left,right)


colors = create_coloring(grid) .|> (x -> Int32.(x)) # convert to Int32 to reduce number of registers



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


"""
    get_local_sides(e_color,n_basefuncs)

Get the local stiffness matrix and force vector for the element `e_color` in the grid.
"""
@inline function get_local_sides(ke_colors, fe_colors,e_color,n_basefuncs)
    start_row_index = (e_color-1)*n_basefuncs + 1
    end_row_index = start_row_index+(n_basefuncs-1)
    start_row_index, end_row_index
    return @view ke_colors[start_row_index:end_row_index,1:n_basefuncs], @view fe_colors[start_row_index:end_row_index]
end


#=NVTX.@annotate=# function assemble_element_gpu!(ke_colors,fe_colors,assembler,cv,dh,n_cells_colored, eles_colored)
    tx = threadIdx().x
    bx = blockIdx().x
    bd = blockDim().x
    # e_color is the global index of the thread in the GPU scheme.
    e_color = tx + (bx-Int32(1))*bd # element number per color

    e_color ≤ n_cells_colored || return nothing
    n_basefuncs = getnbasefunctions(cv)
    # e is the global index of the finite element in the grid.
    e = eles_colored[e_color]
    cell_coords = getcoordinates(dh.grid, e)

    ke,fe = get_local_sides(ke_colors, fe_colors,e_color,n_basefuncs)
    fill!(ke, 0.0f0)
    fill!(fe, 0.0f0)
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

    ## Assemble Ke into Kgpu ##
    assemble!(assembler, celldofs(dh,e),ke,fe)

    return nothing
end


function allocate_local_matrices(colors,cv)
    max_color_size = colors .|> length |> maximum
    n_basefuncs = getnbasefunctions(cv)
    cols = n_basefuncs
    rows = n_basefuncs * max_color_size
    # allocate maximum possible memory to incorporate all local matrices for each color.
    # It will be mutated in each color.
    ke = CUDA.zeros(Float32, rows, cols)
    fe = CUDA.zeros(Float32, rows)
    return ke,fe
end


Adapt.@adapt_structure Ferrite.GPUGrid
Adapt.@adapt_structure Ferrite.GPUDofHandler
Adapt.@adapt_structure Ferrite.GPUAssemblerSparsityPattern

#=NVTX.@annotate=# function assemble_global_gpu_color(cellvalues,dh,colors)
    ke_colors,fe_colors = allocate_local_matrices(colors,cellvalues)
    K = allocate_matrix(SparseMatrixCSC{Float32, Int32},dh)
    Kgpu = CUSPARSE.CuSparseMatrixCSC(K)
    fgpu = CUDA.zeros(ndofs(dh))
    assembler = start_assemble(Kgpu, fgpu)
    n_colors = length(colors)
    # set up kernel adaption & launch the kernel
    dh_gpu = Adapt.adapt_structure(CuArray, dh)
    assembler_gpu = Adapt.adapt_structure(CUDA.KernelAdaptor(), assembler)
    cellvalues_gpu = Adapt.adapt_structure(CuArray, cellvalues)
    for i in 1:n_colors
        kernel = @cuda launch=false assemble_element_gpu!(ke_colors,fe_colors,assembler_gpu,cellvalues_gpu,dh_gpu,Int32(length(colors[i])),cu(colors[i]))
        #@show CUDA.registers(kernel)
        config = launch_configuration(kernel.fun)
        threads = min(length(colors[i]), config.threads)
        blocks =  cld(length(colors[i]), threads)
        kernel(ke_colors,fe_colors,assembler_gpu,cellvalues,dh_gpu,Int32(length(colors[i])),cu(colors[i]);  threads, blocks)
    end
    return Kgpu,fgpu
end



stassy(cv,dh) = assemble_global!(cv,dh,Val(false))




# qpassy(cv,dh) = assemble_global!(cv,dh,Val(true))


Kgpu, fgpu =  assemble_global_gpu_color(cellvalues,dh,colors)
#Kgpu, fgpu = CUDA.@profile    assemble_global_gpu_color(cellvalues,dh,colors)
# to benchmark the code using nsight compute use the following command: ncu --mode=launch julia
# Open nsight compute and attach the profiler to the julia instance
# ref: https://cuda.juliagpu.org/v2.2/development/profiling/#NVIDIA-Nsight-Compute
# to benchmark using nsight system use the following command: # nsys profile --trace=nvtx julia rmse_kernel_v1.jl


norm(Kgpu)



Kstd , Fstd =  stassy(cellvalues,dh);
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
        Kgpu, fgpu =  assemble_global_gpu_color(cellvalues,dh,colors)

        @test norm(Kstd) ≈ norm(Kgpu) atol=1e-4
    end
end
