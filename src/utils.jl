using Preferences

const DEBUG = @load_preference("use_debug", false)

"""
toggle_debug(use_debug = true)

Helper to turn on or off debug helpers in Ferrite.
"""
function toggle_debug(use_debug = true)
    @set_preferences!("use_debug" => use_debug)
    DEBUG_STR = DEBUG ? "ON" : "OFF"
    USE_DEBUG_STR = use_debug ? "ON" : "OFF"
    @info("Toggling debug mode from $DEBUG_STR to $USE_DEBUG_STR. Restart your Julia session for this change to take effect!")
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
