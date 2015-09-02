"""
Computes the element stiffness matrix *Ke* for a
spring element.
"""
function spring1e(k::Float64)
    Ke = [k -k;
          -k k]
end
spring1e(k::Number) = spring1e(convert(Float64, k))


"""
Computes the force *fe* for a spring element
"""
function spring1s(k::Float64, u::Vector{Float64})
    if length(u) != 2
        throw(ArgumentError("displacements for computing the spring force must" *
                            "be a vector of length 2"))
    end
    return k * (u[2] - u[1])
end
spring1s{T<:Number, P<:Number}(k::T, u::P) = spring1s(convert(Float64, k),
                                                convert(Vector{Float64}, u))


