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
    return
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

convert_to_orderedset(set::AbstractVector{T}) where {T} = OrderedSet{T}(set)
convert_to_orderedset(set::AbstractSet{T}) where {T} = convert(OrderedSet{T}, set)

function convert_to_orderedsets(namedsets::Dict{String, <:AbstractVecOrSet{T}}) where {T}
    return Dict{String, OrderedSet{T}}(k => convert_to_orderedset(v) for (k, v) in namedsets)
end
convert_to_orderedsets(namedsets::Dict{String, <:OrderedSet}) = namedsets

"""
    asset_url(file::AbstractString)

Get the url for the asset `file`. Please note that no promises are made that files that currently exists will be kept.
This function is only used to facilitate downloading assets for tutorials etc, allowing a centrally
administrated locations for all assets.

Only Ferrite maintainers can add assets, which are added by directly uploading to the
[`assets` folder in the `gh-pages` branch](https://github.com/Ferrite-FEM/Ferrite.jl/tree/gh-pages/assets).
"""
asset_url(file::AbstractString) = string("https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/", file)
