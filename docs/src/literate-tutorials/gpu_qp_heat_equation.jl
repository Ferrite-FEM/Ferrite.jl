using Ferrite, CUDA

left = Tensor{1,2,Float64}((0,-0)) # define the left bottom corner of the grid.
right = Tensor{1,2,Float64}((400.0,400.0)) # define the right top corner of the grid.
grid = generate_grid(Quadrilateral, (100, 100),left,right); 


ip = Lagrange{RefQuadrilateral, 1}() # define the interpolation function (i.e. Bilinear lagrange)

# define the numerical integration rule 
# (i.e. integrating over quad shape with two quadrature points per direction)
qr = QuadratureRule{RefQuadrilateral}(2) 
cellvalues = CellValues(qr, ip);


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
function assemble_element_qpiter!(Ke::Matrix, fe::Vector, cellvalues,cell_coords::AbstractVector)
    n_basefuncs = getnbasefunctions(cellvalues)
    ## Loop over quadrature points
    for qv in Ferrite.QuadratureValuesIterator(cellvalues,cell_coords)
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
K = create_sparsity_pattern(dh)

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


function assemble_element_gpu!(Kgpu,cv,dh) 
    i = threadIdx().x 
    j = threadIdx().y 
    bx = blockIdx().x
    cell_coords = getcoordinates(dh.grid)
    n_basefuncs = getnbasefunctions(cv)
    #Ke = CuStaticSharedArray(Float32, (n_basefuncs, n_basefuncs)) # We don't need shared memory
    keij = 0.0
    for qv in Ferrite.QuadratureValuesIterator(cv,cell_coords) 
        # Get the quadrature weight
        dΩ = getdetJdV(qv)
        ## Get test function gradient
        ∇δu = shape_gradient(qv, i)
        ## Get shape function gradient
        ∇u = shape_gradient(qv, j)

        keij += (∇δu ⋅ ∇u) * dΩ 
    end 
    #Ke[i,j] = keij # We don't need shared memory

    # TODO: Assemble local matrix here in Kgpu
    # TODO: Add abstraction, Addittionally use assembler to assemble the local matrix into global matrix.
    dofs = dh.cell_dofs
    @inbounds ig = dofs[(bx-1)*n_basefuncs+i]
    @inbounds jg = dofs[(bx-1)*n_basefuncs+j] 
    CUDA.@atomic Kgpu[ig, jg] += keij
    return nothing
end


function assemble_global_gpu(cellvalues,dh)
    Kgpu =   CUDA.zeros(dh.ndofs.x,dh.ndofs.x)
    n_base_funcs = getnbasefunctions(cellvalues) 
    # each block represents a cell, and every (i,j) in the 2D threads represents an element in the local stiffness matrix. 
    @cuda blocks=length(dh.grid.cells) threads = (n_base_funcs,n_base_funcs) assemble_element_gpu!(Kgpu,cellvalues,dh)
    return Kgpu
end




stassy(cv,dh) = assemble_global!(cv,dh,Val(false))


qpassy(cv,dh) = assemble_global!(cv,dh,Val(true))


using BenchmarkTools
using LinearAlgebra


Kgpu =@btime assemble_global_gpu($cellvalues,$dh);

 
norm(Kgpu)

Kstd , Fstd = @btime stassy($cellvalues,$dh);

norm(Kstd)


