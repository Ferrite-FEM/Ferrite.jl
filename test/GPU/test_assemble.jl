# Helper function to initialize sparse matrices
function init_sparse_matrix(n, m)
    row_indices = Int32[]
    col_indices = Int32[]
    values = Float32[]
    for i in 1:min(n, m)
        push!(row_indices, i)
        push!(col_indices, i)
        push!(values, Float32(0.0))
    end
    sparse_matrix = sparse(row_indices, col_indices, values, n, m)
    return sparse_matrix
end


function assemble_kernel!(K, f, dofs, Ke, fe)
    # kernel that only assembles local into global.
    A = start_assemble(K, f)
    assemble!(A, dofs, Ke, fe)
end

# Test for assembling global stiffness matrix and force vector
@testset "Test assemble!" begin
    # System parameters
    n = 5
    m = 5
    dofs = Int32[1, 2, 3, 4, 5]
    Ke = CUDA.fill(1.0f0, n, m)  # Local stiffness matrix (shared memory)
    fe = CUDA.fill(1.0f0, n)     # Local force vector (shared memory)

    # Initialize global stiffness matrix and global force vector
    K = CUSPARSE.CuSparseMatrixCSC(init_sparse_matrix(n, m))

    f = CUDA.fill(0.0f0, n)

    @cuda blocks = 5 threads = 5 assemble_kernel!(K, f, cu(dofs), Ke, fe) # 5 * 5 = 25 threads

    # Test: Check force vector update
    @test all(f .≈ CUDA.fill(25.0f0, n))
    # Test: Check global stiffness matrix update (values must be incremented by 25 = 5 * 5)
    @test all(K.nzVal .≈ CUDA.fill(25.0f0, length(K.nzVal)))

end
