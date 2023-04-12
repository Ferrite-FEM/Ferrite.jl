import Preferences

const DEBUG = Preferences.@load_preference("use_debug", false)

"""
    Ferrite.debug_mode(; enable=true)

Helper to turn on (`enable=true`) or off (`enable=false`) debug expressions in Ferrite.

Debug mode influences `Ferrite.@debug expr`: when debug mode is enabled, `expr` is
evaluated, and when debug mode is disabled `expr` is ignored.
"""
function debug_mode(; enable = true)
    if DEBUG == enable == true
        @info "Debug mode already enabled."
    elseif DEBUG == enable == false
        @info "Debug mode already disabled."
    else
        Preferences.@set_preferences!("use_debug" => enable)
        @info "Debug mode $(enable ? "en" : "dis")abled. Restart the Julia session for this change to take effect!"
    end
end

@static if DEBUG
    @eval begin
        macro debug(ex)
            return :($(esc(ex)))
        end
    end
else
     @eval begin
        macro debug(ex)
            return nothing
        end
    end
end

mutable struct ScalarWrapper{T}
    x::T
end

@inline Base.getindex(s::ScalarWrapper) = s.x
@inline Base.setindex!(s::ScalarWrapper, v) = s.x = v
Base.copy(s::ScalarWrapper{T}) where {T} = ScalarWrapper{T}(copy(s.x))

# Cast to Tensors.(Vec|Tensor) when possible
for dim in 1:3 @eval begin
    @inline function tensor_cast(v::SVector{$dim,T}) where T
        return Vec{$dim,T}(Tuple(v))
    end
    @inline function tensor_cast(v::SMatrix{$dim,$dim,T,M}) where {T,M}
        return Tensor{2,$dim,T,M}(Tuple(v))
    end
end end

@inline tensor_cast(x) = x # Fallback
