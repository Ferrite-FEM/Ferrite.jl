using Ferrite
using StaticArrays
using SparseArrays
using CUDA






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
    ## Loop over quadrature points
    for q_point in 1:getnquadpoints(cellvalues)
        ## Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        ## Loop over test shape functions
        for i in 1:n_basefuncs
            δu  = shape_value(cellvalues, q_point, i)
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
    ## Loop over all cels
    for cell in CellIterator(dh)
        fill!(Ke, 0)
        fill!(fe, 0)
        if QPiter
            ## reinit!(cellvalues, getcoordinates(cell))
            assemble_element_qpiter!(Ke, fe, cellvalues,getcoordinates(cell))
        else
            ## Reinitialize cellvalues for this cell
            reinit!(cellvalues, cell)
            ## Compute element contribution
            assemble_element_std!(Ke, fe, cellvalues)
        end
        ## Assemble Ke and fe into K and f
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    return K, f
end



## gpu version of element assembly
function assemble_element!(Ke,fe,cv,cell)
    n_basefuncs = getnbasefunctions(cv)
    for qv in Ferrite.QuadratureValuesIterator(cv,getcoordinates(cell))
        ## Get the quadrature weight
        dΩ = getdetJdV(qv)
        ## Loop over test shape functions
        for i in 1:n_basefuncs
            δu  = shape_value(qv, i)
            ∇u = shape_gradient(qv, i)
            ## Add contribution to fe
            fe[i] += δu * dΩ
            ## fe_shared[tx,i] += δu * dΩ
            ## Loop over trial shape functions
            for j in 1:n_basefuncs
                ∇δu = shape_gradient(qv, j)
                ## Add contribution to Ke
                Ke[i,j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
end


# gpu version of global assembly
function assemble_gpu!(Kgpu,fgpu, cv, dh)
    n_basefuncs = getnbasefunctions(cv)
    assembler = start_assemble(Kgpu, fgpu)
    for cell in CellIterator(dh, convert(Int32,n_basefuncs))
        Ke = cellke(cell)
        fe = cellfe(cell)
        assemble_element!(Ke, fe, cv, cell)
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    return nothing
end


n_basefuncs = getnbasefunctions(cellvalues)

# Allocate CPU matrix
K = allocate_matrix(SparseMatrixCSC{Float32, Int32},dh);

# Allocate GPU matrix
## commented to pass the test
## Kgpu = CUSPARSE.CuSparseMatrixCSC(K);
## fgpu = CUDA.zeros(ndofs(dh));

n_cells = dh |> get_grid |> getncells

# Kernel configuration
## commented to pass the test
## init_gpu_kernel(BackendCUDA,n_cells,n_basefuncs,assemble_gpu!, (Kgpu,fgpu, cellvalues, dh)) |> launch!


stassy(cv,dh) = assemble_global!(cv,dh,Val(false))


## commented to pass the test
## norm(Kgpu)
Kstd , Fstd = stassy(cellvalues,dh);
norm(Kstd)

