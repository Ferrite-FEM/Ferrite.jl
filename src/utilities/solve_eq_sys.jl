"""
    solveq(K, f, bc, [symmetric=false]) -> a, fb

Solves the equation system Ka = f taking into account the
Dirichlet boundary conditions in the matrix `bc`. Returns the solution vector `a`
and reaction forces `fb`
If `symmetric` is set to `true`, the matrix will be factorized with Cholesky factorization.
"""
function solveq(K::AbstractMatrix, f::Array, bc::Matrix, symmetric=false)
    if size(K, 1) != size(K, 2)
        throw(DimensionMismatch("matrix need to be square"))
    end
    n = size(K, 2)
    nrf = length(f)
    if n != nrf
        throw(DimensionMismatch("Mismatch between number of rows in the stiffness matrix and load vector (#rowsK=$n #rowsf=$nrf)"))
    end

    d_pres = convert(Vector{Int}, bc[:,1])   # prescribed dofs
    a_pres = bc[:,2] # corresponding prescribed dof values

    # Construct array holding all the free dofs
    d_free = setdiff(collect(1:n), d_pres)

    # Solve equation system and create full solution vector a
    if symmetric
        K_fact = cholfact(Symmetric(K[d_free, d_free], :U))
        a_free = K_fact \ (f[d_free] - K[d_free, d_pres] * a_pres)
    else
        a_free = K[d_free, d_free] \ (f[d_free] - K[d_free, d_pres] * a_pres)
    end
    a = zeros(n)
    a[d_free] = a_free
    a[d_pres] = a_pres

    # Compute boundary force = reaction force
    f_b = K*a - f;

    return a, f_b
end


function solveq(K::AbstractMatrix, f::Array, symmetric=false)
    n = chksquare(K)
    nrf = length(f)
    if n != nrf
        throw(DimensionMismatch("Mismatch between number of rows in the stiffness matrix and load vector (#rowsK=$n #rowsf=$nrf)"))
    end

    # Solve equation system and create full solution vector a
    if symmetric
        K_fact = cholfact(Symmetric(K, :U))
        a = K_fact \ f
    else
        a = K \ f
    end

    return a
end
