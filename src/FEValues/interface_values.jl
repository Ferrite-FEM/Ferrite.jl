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
shape_value(iv::InterfaceValues, qp::Int, base_func::Int, here::Bool) = f_value_(iv, qp, base_func, here, shape_value) 
shape_value_jump(iv::InterfaceValues, qp::Int, base_func::Int) = f_jump_(iv, qp, base_func, shape_value) 
shape_value_average(iv::InterfaceValues, qp::Int, base_func::Int) = f_average(iv, qp, base_func, shape_value) 

shape_gradient(iv::InterfaceValues, qp::Int, base_func::Int, here::Bool) = f_value_(iv, qp, base_func, here, shape_gradient) 
shape_gradient_jump(iv::InterfaceValues, qp::Int, base_func::Int) = f_jump_(iv, qp, base_func, shape_gradient) 
shape_gradient_average(iv::InterfaceValues, qp::Int, base_func::Int) = f_average(iv, qp, base_func, shape_gradient) 

function f_value_(iv::InterfaceValues, qp::Int, base_func::Int, here::Bool, f_::Function)
    dofs = iv.dofmap[base_func]
    here && !isnothing(dofs[1]) && return f_(iv.face_values, qp, base_func)
    !here && !isnothing(dofs[2]) && return f_(iv.face_values_neighbor, qp, base_func)
    return 0.0
end
function f_jump_(iv::InterfaceValues, qp::Int, base_func::Int, f_::Function)
    dofs = iv.dofmap[base_func]
    jump = 0.0
    jump += !isnothing(dofs[1]) ? f_(iv.face_values, qp, base_func) ⋅ getnormal(iv.face_values, qp) : 0
    jump += !isnothing(dofs[2]) ?  f_(iv.face_values_neighbor, qp, base_func) ⋅ getnormal(iv.face_values_neighbor, qp) : 0
    return jump
end
function f_average_(iv::InterfaceValues, qp::Int, base_func::Int, f_::Function)
    dofs = iv.dofmap[base_func]
    average = 0.0
    average += !isnothing(dofs[1]) ?  0.5 * f_(iv.face_values, qp, base_func) : 0
    average += !isnothing(dofs[2]) ?  0.5 * f_(iv.face_values_neighbor, qp, base_func) : 0
    return average
end