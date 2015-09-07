"""
Solves the equation system Ka = f taking into account the Dirichlet boundary conditions in the matrix bc
"""
function solve_eq_sys(K::AbstractMatrix, f::Array, bc::Matrix )
    n = chksquare(K)
    nrf = length(f)
    if n != nrf
        throw(DimensionMismatch("Mismatch between number of rows in the stiffness matrix and load vector (#rowsK=$n #rowsf=$nrf)"))
    end

    if isa(K, SparseMatrixCSC)
         _K = convert(SparseMatrixCSC{Float64}, K)
    else
        _K = convert(Matrix{Float64}, K)
    end
    d_pres = convert(Vector{Int}, bc[:,1])   # prescribed dofs
    a_pres = convert(Vector{Float64}, bc[:,2])   # corresponding prescribed dof values

    # Construct array holding all the free dofs
    d_free = setdiff(collect(1:n), d_pres)

    # Solve equation system and create full solution vector a
    a_free = K[d_free, d_free] \ (f[d_free] - K[d_free, d_pres] * a_pres)
    a = zeros(n)
    a[d_free] = a_free
    a[d_pres] = a_pres

    # Compute boundary force = reaction force
    f_b = K*a - f;

    return a, f_b

end

const solveq = solve_eq_sys # for CALFEM API compatability
