# Defines InterfaceScalarValues and InterfaceVectorValues and common methods
# InterfaceValues
struct InterfaceValues{FaceValuesType<:FaceValues}
    n_quadrature_points::ScalarWrapper{Int}
    interface_dof_indices::Vector{Int}
    dofmap::Vector{Array{Int,1}}
    face_values :: FaceValuesType
    face_values_neighbor :: FaceValuesType
end
function InterfaceValues(quad_rule::QuadratureRule, func_interpol::Interpolation,
    geom_interpol::Interpolation=func_interpol)
    face_values = FaceScalarValues(quad_rule, func_interpol, geom_interpol)
    face_values_neighbor = FaceScalarValues(quad_rule, func_interpol, geom_interpol)
    InterfaceValues{FaceScalarValues}(ScalarWrapper(0),[],[],face_values,face_values_neighbor)
end
function reinit!(iv::InterfaceValues, dh::AbstractDofHandler, cidx::Int, coords::AbstractVector{Vec{dim,T}}, f::Int, ncidx::Int, ncoords::AbstractVector{Vec{dim,T}}, nf::Int) where {dim,T}
    reinit!(iv.face_values,coords,f)
    reinit!(iv.face_values_neighbor,ncoords,nf)
    @assert getnquadpoints(iv.face_values) == getnquadpoints(iv.face_values_neighbor)
    iv.n_quadrature_points[] = getnquadpoints(iv.face_values)
    v = celldofs(dh,cidx)
    v2 = celldofs(dh,ncidx)
    temp_dict = Dict{Int,Array{Union{Int, Nothing},1}}()
    for (local_dof, global_dof) in enumerate(v)
        !haskey(temp_dict,global_dof) && push!(temp_dict,global_dof=>Array{Union{Int, Nothing},1}(nothing,2))
        temp_dict[global_dof][1] = local_dof
    end
    for (local_dof, global_dof) in enumerate(v2)
        !haskey(temp_dict,global_dof) && push!(temp_dict,global_dof=>Array{Union{Int, Nothing},1}(nothing,2))
        temp_dict[global_dof][2] = local_dof
    end
    resize!(iv.dofmap,length(temp_dict))
    resize!(iv.interface_dof_indices,length(temp_dict))
    for (idx,dofs_pair) in enumerate(temp_dict)
        iv.interface_dof_indices[idx] = dofs_pair[1]
        iv.dofmap[idx] = dofs_pair[2]
    end
end
function shape_value(iv::InterfaceValues, qp::Int, base_func::Int, here::Bool)
    dof_pair = iv.dofmap[base_func]
    here && !isnothing(dof_pair[1]) && return shape_value(iv.face_values, qp, base_func)
    !here && !isnothing(dof_pair[2]) && return shape_value(iv.face_values_neighbor, qp, base_func)
    return 0.0
end
function shape_value(iv::InterfaceValues, qp::Int, base_func::Int, here::Bool)
    dof_pair = iv.dofmap[base_func]
    here && !isnothing(dof_pair[1]) && return shape_value(iv.face_values, qp, base_func)
    !here && !isnothing(dof_pair[2]) && return shape_value(iv.face_values_neighbor, qp, base_func)
    return 0.0
end