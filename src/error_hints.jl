# Hints printed together with errors for common user mistakes. The hints are registered in
# `__init__` since `register_error_hint` modifies global state in Base, which is not
# persisted by precompilation.

# Innermost element type of (possibly nested) vectors and dicts, e.g. Vec{3} for
# Vector{Vector{Vec{3}}} and Dict{Int, Vector{Vec{3}}}
function _innermost_eltype(@nospecialize(T))
    while true
        if T isa Type && T <: Union{Tensor, SymmetricTensor} # Tensors are AbstractArrays too
            return T
        elseif T isa Type && T <: AbstractArray
            T = eltype(T)
        elseif T isa Type && T <: AbstractDict
            T = valtype(T)
        else
            return T
        end
    end
    return
end

# Data with a non-concrete tensor element type (e.g. created by pushing into `Vec{3}[]`) is not
# supported by e.g. `project` and `write_node_data`, since `Vec{3}` is not an `AbstractTensor`.
function _nonconcrete_tensor_eltype_hint(io::IO, exc::MethodError, argtypes, kwargs)
    exc.f === project || exc.f === write_node_data || return
    for argtype in argtypes
        E = _innermost_eltype(argtype)
        if E isa Type && E <: Union{Tensor, SymmetricTensor} && !isconcretetype(E)
            print(io, "\nThe element type `", E, "` of the data is not concrete. Use a concrete ")
            E64 = E isa UnionAll ? E{Float64} : nothing
            if E64 !== nothing && isconcretetype(E64)
                print(io, "element type such as `", E64, "` instead, e.g. `", E64, "[]` instead of `", E, "[]`.")
            else
                print(io, "element type, e.g. `Vec{3, Float64}` instead of `Vec{3}`.")
            end
            return
        end
    end
    return
end

function __init__()
    Base.Experimental.register_error_hint(_nonconcrete_tensor_eltype_hint, MethodError)
    return nothing
end
