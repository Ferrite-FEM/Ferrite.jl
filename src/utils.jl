using Preferences

const DEBUG = @load_preference("use_debug", false)

"""
debug_mode(enable = true)

Helper to turn on or off debug helpers in Ferrite.
"""
function debug_mode(enable = true)
    @set_preferences!("use_debug" => enable)
    DEBUG_STR = DEBUG ? "ON" : "OFF"
    NEW_DEBUG_STR = enable ? "ON" : "OFF"
    if DEBUG != enable
        @info("Toggling debug mode from $DEBUG_STR to $NEW_DEBUG_STR. Restart your Julia session for this change to take effect!")
    else
        @info("Debug mode already $DEBUG_STR.")
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
