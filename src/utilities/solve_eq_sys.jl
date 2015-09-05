"""
Solves the equation system Ka = f taking into account the Dirichlet boundary conditions in the matrix bc
"""
function solve_eq_sys(K::AbstractMatrix, f::Array, bc::Matrix )

    (nr, nc) = size(K);
    if nr != nc
        throw(DimensionMismatch("Stiffness matrix is not square (#rows=%nr #cols=$nc)"))
    elseif nr != length(f)
        throw(DimensionMismatch("Mismatch between number of rows in the stiffness matrix and load vector (#rowsK=$nr #rowsf=$length(f))"))
    end

    _Ke = convert(Matrix{Float64}, Ke)
    d_pres = bc[:,1]   # prescribed dofs
    a_pres = bc[:,2]   # corresponding prescribed dof values

    _d_pres = convert(Array{Int}, d_pres)
    _a_pres = convert(Array{Float64}, a_pres)

    # Construct array holding all the free dofs
    d_free = [1:nr]
    deleteat!(d_free, _d_pres)

    # Solve equation system and create full solution vector a
    a_free = K[d_free, d_free] \ ( f[d_free] - K[d_free, d_pres] * a_pres)
    a = zeros(nr)
    a[d_free] = a_free
    a[d_pres] = _a_pres

    # Compute boundary force = reaction force
    f_b = K*a - f;

    return a, f_b

end

const solveq = solve_eq_sys # for CALFEM API compatability