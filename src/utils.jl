type ScalarWrapper{T}
    x::T
end

Base.getindex(s::ScalarWrapper) = s.x
Base.setindex!(s::ScalarWrapper, v) = s.x = v
