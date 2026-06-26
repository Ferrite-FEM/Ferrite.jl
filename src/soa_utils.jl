# ------------------------------ User-facing part -------------------------------
"""
    distribute_to_tasks(backend, obj, num_tasks)

Distribute the object `obj` to `num_tasks` based on the chosen backend for task-based paralellism.
Returns `d::AbstractVector{T}` where `T` is loosely equivalent to `typeof(obj)`, meaning that most
methods are applicable to both types. Supported `backend`s are

"""
function distribute_to_tasks end

struct SoAContainer{T, T_inner} <: AbstractVector{T}
    soa::T_inner
    nels::Int
    function SoAContainer(soa::T_inner, nels::Int) where {T_inner}
        #TODO: Check bounds here so we can guarantee inbounds later?
        T = typeof(get_substruct(1, soa))
        return new{T, T_inner}(soa, nels)
    end
end

Base.size(c::SoAContainer) = (c.nels,)
Base.getindex(c::SoAContainer, i::Integer) = get_substruct(i, c.soa) # TODO: Why reverse indexing here???
function Base.show(io::IO, d::MIME"text/plain", c::SoAContainer{T}) where {T}
    println(io, "SoAContainer{$T}")
    print(io, "Structure of Arrays container with $(c.nels) elements.")
    #= # Requires GPUArraysCore dependency
    println(io, " First element:")
    GPUArraysCore.allowscalar() do
        show(io, d, c[1])
    end
    =#
end

# ------------------------------ Internal part --------------------------------------
view_from_shared(::Nothing, i::Integer) = nothing
view_from_shared(a::AbstractArray{<:Any, 1}, i::Integer) = view(a, i:i)
view_from_shared(a::AbstractArray{<:Any, 2}, i::Integer) = view(a, i, :)
view_from_shared(a::AbstractArray{<:Any, 3}, i::Integer) = view(a, i, :, :)

# Extract the i-th worker's local slice from batched device data
function get_substruct(i, cv::CellValues)
    fv = get_substruct(i, cv.fun_values)
    return CellValues(fv, cv.geo_mapping, cv.qr, view_from_shared(cv.detJdV, i))
end

function get_substruct(i, fv::FunctionValues)
    Nx = fv.Nξ === fv.Nx ? fv.Nx : view_from_shared(fv.Nx, i)
    dNdx = view_from_shared(fv.dNdx, i)
    d2Ndx2 = view_from_shared(fv.d2Ndx2, i)
    return FunctionValues(fv.ip, Nx, fv.Nξ, dNdx, fv.dNdξ, d2Ndx2, fv.d2Ndξ2)
end

function get_substruct(i, cc::CellCache)
    return CellCache(
        cc.flags, cc.grid, view_from_shared(cc.cellid, i),
        view_from_shared(cc.nodes, i), view_from_shared(cc.coords, i),
        cc.dh, view_from_shared(cc.dofs, i)
    )
end
