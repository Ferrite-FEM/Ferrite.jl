type ScalarWrapper{T}
    x::T
end

@inline Base.getindex(s::ScalarWrapper) = s.x
@inline Base.setindex!(s::ScalarWrapper, v) = s.x = v

function lagrange_polynomial{dim, T}(x::Vec{dim, T}, xs::AbstractVector, j::Int)
    v = one(T)
    for i in 1:dim
        v *= lagrange_polynomial(x[i], xs, j)
    end
    return v
end

# http://math.stackexchange.com/a/809946
function lagrange_polynomial{dim}(x::Number, xs::AbstractVector, j::Int)
    @assert j <= length(xs)
    num, den = one(x), one(x)
    @inbounds for i in 1:length(xs)
        i == j && continue
        num *= (x - xs[i])
        den *= (xs[i] - xs[j])
    end
    return num / den
end