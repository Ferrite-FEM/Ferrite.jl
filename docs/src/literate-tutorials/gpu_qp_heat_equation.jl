using Ferrite, CUDA



left = Tensor{1,2,Float32}((0,-0)) # define the left bottom corner of the grid.
right = Tensor{1,2,Float32}((100.0,100.0)) # define the right top corner of the grid.

grid = generate_grid(Quadrilateral, (100, 100),left,right); 


ip = Lagrange{RefQuadrilateral, 1}() # define the interpolation function (i.e. Bilinear lagrange)

# define the numerical integration rule 
# (i.e. integrating over quad shape with two quadrature points per direction)
qr = QuadratureRule{RefQuadrilateral,Float32}(2) 
cellvalues = CellValues(Float32,qr, ip);


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
# function assemble_element_qpiter!(Ke::Matrix, fe::Vector, cellvalues,cell_coords::AbstractVector)
#     n_basefuncs = getnbasefunctions(cellvalues)
#     ## Loop over quadrature points
#     for qv in Ferrite.QuadratureValuesIterator(cellvalues,cell_coords)
#         ## Get the quadrature weight
#          dΩ = getdetJdV(qv)
#         ## Loop over test shape functions
#         for i in 1:n_basefuncs
#             δu  = shape_value(qv, i)
#             ∇δu = shape_gradient(qv, i)
#             ## Add contribution to fe
#             fe[i] += δu * dΩ
#             ## Loop over trial shape functions
#             for j in 1:n_basefuncs
#                 ∇u = shape_gradient(qv, j)
#                 ## Add contribution to Ke
#                 Ke[i, j] += (∇δu ⋅ ∇u) * dΩ
#             end
#         end
#     end
#     return Ke, fe
# end

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



### Old impelementation that makes each threads over the quadrature points.
# function assemble_element_gpu!(Kgpu,cv,dh) 
#     i = threadIdx().x 
#     j = threadIdx().y 
#     bx = threadIdx().z + (blockIdx().x - Int32(1)) * blockDim().z # element number
#     bx <= length(dh.grid.cells) || return nothing
#     #bx = blockIdx().x
#     cell_coords = getcoordinates(dh.grid)
#     n_basefuncs = Int32(getnbasefunctions(cv))
#     #Ke = CuStaticSharedArray(Float32, (n_basefuncs, n_basefuncs)) # We don't need shared memory
#     keij = 0.0f0
#     for qv in Ferrite.QuadratureValuesIterator(cv,cell_coords) 
#         # Get the quadrature weight
#         dΩ = getdetJdV(qv)
#         ## Get test function gradient
#         ∇δu = shape_gradient(qv, i)
#         ## Get shape function gradient
#         ∇u = shape_gradient(qv, j)

#         keij += (∇δu ⋅ ∇u) * dΩ 
#     end 
#     #Ke[i,j] = keij # We don't need shared memory
    
#     # TODO: Assemble local matrix here in Kgpu
#     # TODO: Add abstraction, Addittionally use assembler to assemble the local matrix into global matrix.
#     dofs = dh.cell_dofs
#     @inbounds ig = Int32(dofs[Int32((bx-Int32(1))*n_basefuncs+i)])
#     @inbounds jg = Int32(dofs[Int32((bx-Int32(1))*n_basefuncs+j)] )

#     ## Sparse Addition ##
#     col_start = Kgpu.colptr[jg]
#     col_end = Kgpu.colptr[jg + 1] - 1

#     for k in col_start:col_end
#         if Kgpu.rowval[k] == ig
#             # Update the existing element
#             CUDA.@atomic Kgpu.nzval[k] += keij
#             return
#         end
#     end
#     ##custom_atomic_add!(Kgpu, keij, ig, jg)
#     #Kgpu[ig, jg] += Float32(keij)
     
#     return nothing
# end


function assemble_element_gpu!(Kgpu,cv,dh) 
    i = threadIdx().x 
    j = threadIdx().y
    q_point = threadIdx().z # quadrature point
   
    bx = blockIdx().x # element number

    cell_coords = getcoordinates(dh.grid)
    n_basefuncs = getnbasefunctions(cv)

    Ke = CuStaticSharedArray(Float32, (n_basefuncs, n_basefuncs))
    Ke[i,j] = 0.0f0

    # Get the quadrature point values (object that encapsulates all the values of the shape functions and their gradient at each node of the cell)
    qv = Ferrite.quadrature_point_values(cv, q_point, cell_coords)
   
    # Get the quadrature weight
    dΩ =getdetJdV(qv)
    ## Get test function gradient
    ∇δu =shape_gradient(qv, i)
    ## Get shape function gradient
    ∇u =shape_gradient(qv, j)

    sync_threads()



    CUDA.@atomic Ke[i,j] += (∇δu ⋅ ∇u) * dΩ 

    #Ke[i,j] = keij # We don't need shared memory
    
    
    dofs = dh.cell_dofs
    ig = dofs[(bx-1)*n_basefuncs+i]
    jg = dofs[(bx-1)*n_basefuncs+j] 
    
    
    sync_threads()

    ## Sparse Addition ##
    # q_point == 1 || return nothing

    # col_start = Kgpu.colptr[jg]
    # col_end = Kgpu.colptr[jg + 1] - 1

    # for k in col_start:col_end
    #     if Kgpu.rowval[k] == ig
    #         # Update the existing element
    #         CUDA.@atomic Kgpu.nzval[Int32(k)] += Ke[i,j]
    #         return
    #     end
    # end
     q_point == 1 || return nothing
    CUDA.@atomic Kgpu[ig, jg] += Ke[i,j]
        
     
    return nothing
end

1 ÷ 3


function assemble_global_gpu(cellvalues,dh)
    #Kgpu =   CUDA.zeros(dh.ndofs.x,dh.ndofs.x)
    n_base_funcs = getnbasefunctions(cellvalues) 

    #K = create_sparsity_pattern(dh)
    #Kgpu = GPUSparseMatrixCSC( Int32(K.m), Int32(K.n), cu(Int32.(K.colptr)), cu(Int32.(K.rowval)), cu(Float32.(K.nzval)))
    # each block represents a cell, and every (i,j) in the 2D threads represents an element in the local stiffness matrix. 
    #n_blocks = cld(length(dh.grid.cells), 16) # 16 threads in z direction
    @cuda blocks=length(dh.grid.cells) threads = (n_base_funcs,n_base_funcs,getnquadpoints(cellvalues)) assemble_element_gpu!(Kgpu,cellvalues,dh)
    return Kgpu
end




stassy(cv,dh) = assemble_global!(cv,dh,Val(false))


# qpassy(cv,dh) = assemble_global!(cv,dh,Val(true))


using BenchmarkTools
# using LinearAlgebra


Kgpu = @btime CUDA.@sync   assemble_global_gpu($cellvalues,$dh)



#Kgpu =    assemble_global_gpu(cellvalues,dh)
 
# sqrt(sum(abs2, Kgpu.nzval))
norm(Kgpu)

Kstd , Fstd = @btime stassy($cellvalues,$dh);
Kstd , Fstd = stassy(cellvalues,dh);
# Kstd[2,6]
norm(Kstd)


