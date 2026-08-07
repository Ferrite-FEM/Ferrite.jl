# ------------------------------ User-facing part -------------------------------
"""
    distribute_to_workers(backend, obj, num_workers)

Distribute the object `obj` to `num_workers` based on the chosen backend for task-based parallelism.
Returns `d::AbstractVector{T}` where `T` is loosely equivalent to `typeof(obj)`, meaning that most
methods are applicable to both types.

"""
function distribute_to_workers end

struct SoAContainer{T, T_inner} <: AbstractVector{T}
    soa::T_inner
    nels::Int
    function SoAContainer(soa::T_inner, nels::Int) where {T_inner}
        #TODO: Check bounds here so we can guarantee inbounds later?
        T = typeof(get_substruct(soa, 1))
        return new{T, T_inner}(soa, nels)
    end
end

Base.size(c::SoAContainer) = (c.nels,)
# Assert the element type since e.g. the NamedTuple rebuild in multi-field CellValues
# extraction is not inferable on its own
Base.getindex(c::SoAContainer{T}, i::Integer) where {T} = get_substruct(c.soa, i)::T
function Base.show(io::IO, d::MIME"text/plain", c::SoAContainer{T}) where {T}
    println(io, "SoAContainer{$T}")
    print(io, "Structure of Arrays container with $(c.nels) elements.")
    return nothing
end

# ------------------------------ Internal part --------------------------------------
view_from_shared(::Nothing, i::Integer) = nothing
view_from_shared(a::AbstractArray{<:Any, 1}, i::Integer) = view(a, i:i)
view_from_shared(a::AbstractArray{<:Any, 2}, i::Integer) = view(a, i, :)
view_from_shared(a::AbstractArray{<:Any, 3}, i::Integer) = view(a, i, :, :)

# Extract the i-th worker's local slice from batched device data
function get_substruct(cv::CellValues, i)
    old_fun_values = get_fun_values(cv)
    fun_values = map(fv -> get_substruct(fv, i), old_fun_values)
    fun_values_nt = _rebuild_fun_values_nt(getfield(cv, :fun_values_nt), old_fun_values, fun_values)
    return CellValues(fun_values_nt, fun_values, get_geo_mapping(cv), get_quadrature_rule(cv), view_from_shared(getdetJdVs(cv), i))
end

function get_substruct(fv::FunctionValues, i)
    Nx = fv.Nξ === fv.Nx ? fv.Nx : view_from_shared(fv.Nx, i)
    dNdx = view_from_shared(fv.dNdx, i)
    d2Ndx2 = view_from_shared(fv.d2Ndx2, i)
    return FunctionValues(fv.ip, Nx, fv.Nξ, dNdx, fv.dNdξ, d2Ndx2, fv.d2Ndξ2)
end

function get_substruct(cc::CellCache, i)
    return CellCache(
        cc.flags, cc.grid, view_from_shared(cc.cellid, i),
        view_from_shared(cc.nodes, i), view_from_shared(cc.coords, i),
        cc.dh, view_from_shared(cc.dofs, i)
    )
end
