const DEBUG = false

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

copy!!(x, y) = copy!(resize!(x, length(y)), y)

@static if VERSION < v"0.7.0-DEV.2563"
    const ht_keyindex2! = Base.ht_keyindex2
else
    import Base.ht_keyindex2!
end
