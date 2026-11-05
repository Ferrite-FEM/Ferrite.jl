"""
    _untangle_affine_constraints!(ch::ConstraintHandler)

Untangle the affine constraints in `ch`, i.e. rewrite them such that no master dof of an
affine constraint is itself constrained by an affine constraint. For example, the system

    u1 = u2 + u5
    u2 = u3 + 4 * u10 + 4.0
    u9 = 3 * u2 - 2.0

is tangled since `u2` appears both as a master and a slave dof. After untangling it reads

    u1 = u3 + 4 * u10 + u5 + 4.0
    u2 = u3 + 4 * u10 + 4.0
    u9 = 3 * u3 + 12 * u10 + 10.0

Dirichlet-type master dofs (constraints without coefficients, e.g. `u3 = f3(t)`) are not
substituted here; their contribution to the inhomogeneities is computed in `update!`.

The constraints are viewed as a directed graph where constraint `i` points to constraint
`j` when the slave dof of `j` is a master dof of `i`. The strongly connected components of
this graph are computed with Tarjan's algorithm, which emits them in reverse topological
order. Processing the components in this order means that when a constraint is reached, all
constraints it depends on are already untangled and can be substituted directly, so the
work follows the dependency graph and the coefficient lists produced by substitution.
Only components with a genuine cycle (e.g. `u1 = 2 * u2 + 1, u2 = u1 + 1`) require
solving a linear system; large components use a sparse solve for supported value types.
"""
function _untangle_affine_constraints!(ch::ConstraintHandler{DH, Tv, Ti}) where {DH, Tv, Ti}
    @assert _istangled(ch) "ConstraintHandler is not tangled"
    # Select affine equations once. Originally empty constraints act as Dirichlet
    # masters and remain symbolic; equations that become constant must be substituted.
    rows = findall(_has_coefficients, ch.dofcoefficients)
    slaves = ch.prescribed_dofs[rows]
    coefficients = [ch.dofcoefficients[i]::Vector{Pair{Ti, Tv}} for i in rows]
    constants = Tv[ch.affine_inhomogeneities[i]::Tv for i in rows]
    coefficients, constants = _untangle_affine_constraints(
        slaves, coefficients, constants; nvariables = ndofs(ch.dh)
    )
    for (j, i) in enumerate(rows)
        ch.dofcoefficients[i] = coefficients[j]
        ch.affine_inhomogeneities[i] = constants[j]
        ch.inhomogeneities[i] = constants[j] # effective value, recomputed in update!
    end
    @assert !_istangled(ch)
    return ch
end

# Affine constraints are the ones with (non-empty) coefficients; empty ones act as Dirichlet
_has_coefficients(coeffs) = coeffs !== nothing && !isempty(coeffs)

"""
    _istangled(ch::ConstraintHandler)

Check if the constraint handler has any tangled dofs. An example of a tangled dof is

    u1 = u2 + u5
    u2 = u3 + 4 * u10 + 4.0.

Here, `u2` is a tangled dof as it appears on the left- and right-hand side of the constraints.
"""
function _istangled(ch::ConstraintHandler)
    for coeffs in ch.dofcoefficients
        coeffs === nothing && continue
        for (d, _) in coeffs
            isempty(ch.isconstrained) || ch.isconstrained[d] || continue
            j = get(ch.dofmapping, d, 0)
            j == 0 && continue
            _has_coefficients(ch.dofcoefficients[j]) && return true
        end
    end
    return false
end
