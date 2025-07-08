# Tensor
function _project(proj::L2Projector, qrs_rhs::Vector{<:QuadratureRule}, vars::Union{AbstractVector, AbstractDict}, M::Integer, ::Type{T}) where {T}
    f = zeros(ndofs(proj.dh))
    for (sdh, qr_rhs) in zip(proj.dh.subdofhandlers, qrs_rhs)
        ip_fun = only(sdh.field_interpolations)
        ip_geo = geometric_interpolation(getcelltype(sdh))
        cv = CellValues(qr_rhs, ip_fun, ip_geo; update_gradients = false)
        assemble_tensor_proj_rhs!(f, cv, sdh, vars)
    end

    return proj.M_cholesky \ f
end

function assemble_tensor_proj_rhs!(f::Vector, cellvalues::CellValues, sdh::SubDofHandler, vars::Union{AbstractVector, AbstractDict})
    # Assemble the multi-column rhs, f = ∭( v ⋅ x̂ )dΩ
    n = getnbasefunctions(cellvalues)
    fe = zeros(n)
    nqp = getnquadpoints(cellvalues)

    ## Assemble contributions from each cell
    for cell in CellIterator(sdh)
        fill!(fe, 0)
        cell_vars = vars[cellid(cell)]
        length(cell_vars) == nqp || error("The number of variables per cell doesn't match the number of quadrature points")
        reinit!(cellvalues, cell)

        for q_point in 1:nqp
            dΩ = getdetJdV(cellvalues, q_point)
            qp_vars = cell_vars[q_point]
            for i in 1:n
                v = shape_value(cellvalues, q_point, i)
                fe[i] += (v ⋅ qp_vars) * dΩ
            end
        end
        assemble!(f, celldofs(cell), fe)
    end
    return
end


#=
| L2Projector input | ScalarInterpolation | VectorizedInterpolation | VectorInterpolation |
| Data type input   | - | - | - |
| `Number` | `Number` | `Number` | N/A |
| `Vec` | `Vec` | `Vec` | `Number` |
| `Tensor{2}` | `Tensor{2}` | `Tensor{2}` | N/A |
=#
