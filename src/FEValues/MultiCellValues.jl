"""
    MultiCellValues(;kwargs...)

Create `MultiCellValues` that contains the cellvalues supplied via keyword arguments

```
qr = QuadratureRule{RefTriangle}(2)
ip_vector = Lagrange{RefTriangle,2}()^2
ip_scalar = Lagrange{RefTriangle,1}()
cv_vector = CellValues(qr, ip_vector)
cv_scalar = CellValues(qr, ip_scalar)
cvs = MultiCellValues(;u=cv_vector, p=cv_scalar, T=cv_scalar)
```

`cvs` is `reinit!`:ed as regular cellvalues. 
Functions for getting information about quadrature points and geometric interpolation 
accept `cvs` directly. Functions to access the specific function interpolation values 
are called as `foo(cvs[:u], args...)` for `u`, and equivalent for other keys. 
"""
struct MultiCellValues{CVS<:Tuple,NV<:NamedTuple} <: AbstractCellValues
    values::CVS         # Points only to unique CellValues
    named_values::NV    # Can point to the same CellValues in values multiple times
end
MultiCellValues(;cvs...) = MultiCellValues(NamedTuple(cvs))
function MultiCellValues(named_values::NamedTuple)
    # Extract the unique CellValues checked by ===
    tuple_values = tuple(unique(objectid, values(named_values))...)

    # Check that all values are compatible with eachother
    # allequal julia>=1.8
    cv_ref = first(named_values)
    @assert all( getpoints(cv.qr) ==  getpoints(cv_ref.qr) for cv in tuple_values)
    @assert all(getweights(cv.qr) == getweights(cv_ref.qr) for cv in tuple_values)
    # Note: The following only works while isbitstype(Interpolation)
    @assert all(cv.gip == cv_ref.gip for cv in tuple_values) 

    return MultiCellValues{typeof(tuple_values),typeof(named_values)}(tuple_values, named_values)
end

# Not sure if aggressive constprop is required, but is intended so use to ensure? (Not supported on v1.6)
# Base.@constprop :aggressive Base.getindex(mcv::MultiCellValues, key::Symbol) = getindex(mcv.named_values, key)
Base.getindex(mcv::MultiCellValues, key::Symbol) = getindex(mcv.named_values, key)

# Geometric values should all be equal and hence can be queried from ::MultiCellValues
@propagate_inbounds getngeobasefunctions(mcv::MultiCellValues) = getngeobasefunctions(first(mcv.values))
@propagate_inbounds function geometric_value(mcv::MultiCellValues, q_point::Int, base_func::Int)
    return geometric_value(first(mcv.values), q_point, base_func)
end

# Quadrature
getnquadpoints(mcv::MultiCellValues) = getnquadpoints(first(mcv.values))
# @propagate_inbounds getdetJdV ? 
getdetJdV(mcv::MultiCellValues, q_point::Int) = getdetJdV(first(mcv.values), q_point)

function reinit!(cv::MultiCellValues, x::AbstractVector, args...)
    map(v -> reinit!(v, x, args...), cv.values)
end

function Base.show(io::IO, ::MIME"text/plain", fe_v::MultiCellValues)
    print(io, "MultiCellValues with ", length(fe_v.values), " unique values. Access names:")
    for (name, cv) in pairs(fe_v.named_values)
        println(io)
        print(io, "$name: ")
        show(io, MIME"text/plain"(), cv)
    end
end