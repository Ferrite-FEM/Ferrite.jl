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

# local backport of JuliaLang/julia/#20619
@static if VERSION < v"0.7.0-DEV.601"
    function _groupedunique!(A::AbstractVector)
        isempty(A) && return A
        idxs = eachindex(A)
        y = first(A)
        state = start(idxs)
        i, state = next(idxs, state)
        for x in A
            if !isequal(x, y)
                i, state = next(idxs, state)
                y = A[i] = x
            end
        end
        resize!(A, i - first(idxs) + 1)
    end
else
    import Base._groupedunique!
end
