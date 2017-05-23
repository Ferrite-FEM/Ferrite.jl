type ScalarWrapper{T}
    x::T
end

@inline Base.getindex(s::ScalarWrapper) = s.x
@inline Base.setindex!(s::ScalarWrapper, v) = s.x = v
Base.copy{T}(s::ScalarWrapper{T}) = ScalarWrapper{T}(copy(s.x))
