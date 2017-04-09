"""
An `Interpolation` is used to define shape functions to interpolate
a function between nodes.

**Constructor:**

```julia
Interpolation{dim, reference_shape, order}()
```

**Arguments:**

* `dim`: the dimension the interpolation lives in
* `shape`: a reference shape, see [`AbstractRefShape`](@ref)
* `order`: the highest order term in the polynomial

The following interpolations are implemented:

* `Lagrange{1, RefCube, 1}`
* `Lagrange{1, RefCube, 2}`
* `Lagrange{2, RefCube, 1}`
* `Lagrange{2, RefCube, 2}`
* `Lagrange{2, RefTetrahedron, 1}`
* `Lagrange{2, RefTetrahedron, 2}`
* `Lagrange{3, RefCube, 1}`
* `Serendipity{2, RefCube, 2}`
* `Lagrange{3, RefTetrahedron, 1}`
* `Lagrange{3, RefTetrahedron, 2}`

**Common methods:**

* [`getnbasefunctions`](@ref)
* [`getdim`](@ref)
* [`getrefshape`](@ref)
* [`getorder`](@ref)


**Example:**

```jldoctest
julia> ip = Lagrange{2, RefTetrahedron, 2}()
JuAFEM.Lagrange{2,JuAFEM.RefTetrahedron,2}()

julia> getnbasefunctions(ip)
6
```
"""
@compat abstract type Interpolation{dim, shape, order} end

"""
Returns the dimension of an `Interpolation`
"""
@inline getdim{dim}(ip::Interpolation{dim}) = dim

"""
Returns the reference shape of an `Interpolation`
"""
@inline getrefshape{dim, shape}(ip::Interpolation{dim, shape}) = shape

"""
Returns the polynomial order of the `Interpolation`
"""
@inline getorder{dim, shape, order}(ip::Interpolation{dim, shape, order}) = order

"""
Computes the value of the shape functions at a point ξ for a given interpolation
"""
function value{dim, T}(ip::Interpolation{dim}, ξ::Vec{dim, T})
    value!(ip, zeros(T, getnbasefunctions(ip)), ξ)
end

"""
Computes the gradients of the shape functions at a point ξ for a given interpolation
"""
function derivative{dim, T}(ip::Interpolation{dim}, ξ::Vec{dim, T})
    derivative!(ip, [zero(Tensor{1, dim, T}) for i in 1:getnbasefunctions(ip)], ξ)
end

@inline function checkdim_value{dim}(ip::Interpolation{dim}, N::AbstractVector, ξ::AbstractVector)
    @assert length(ξ) == dim
    n_base = getnbasefunctions(ip)
    length(N) == n_base || throw(ArgumentError("N must have length $(n_base)"))
end

@inline function checkdim_derivative{dim, T}(ip::Interpolation{dim}, dN::AbstractVector{Vec{dim, T}}, ξ::Vec{dim, T})
    n_base = getnbasefunctions(ip)
    length(dN) == n_base || throw(ArgumentError("dN must have length $(n_base)"))
end

function derivative!{T, order, shape, dim}(ip::Interpolation{dim, shape, order}, dN::AbstractVector{Vec{dim, T}}, ξ::Vec{dim, T})
    checkdim_derivative(ip, dN, ξ)
    f!(N, x) = value!(ip, N, x)
    NArray = zeros(T, getnbasefunctions(ip))
    dNArray = ForwardDiff.jacobian(f!, NArray, ξ)
    # Want to use reinterpret but that crashes Julia 0.6, #20847
    for i in 1:length(dN)
        dN[i] = Vec{dim, T}((dNArray[i, :]...))
    end
    return dN
end


"""
Returns the number of base functions for an [`Interpolation`](@ref) or `Values` object.
"""
getnbasefunctions

function create_face_quadrule_for_dofs{dim, refshape}(ip::Interpolation{dim, refshape})
  coords = get_dof_local_coordinates(ip)
  facelist = get_facelist(ip)
  qrs = JuAFEM.QuadratureRule{dim, refshape, Float64}[]
  for f in 1:get_n_faces(ip)
    c = Array(coords[facelist[f]])
    # TODO: FLoat64 -> something?
    push!(qrs, QuadratureRule{dim, refshape, Float64}(ones(length(c)), c))
  end
  return qrs
end

# General geometrical information
@pure get_n_cells(::Interpolation) = 1

get_n_faces(ip::Interpolation{1}) = get_n_vertices(ip)
get_n_faces(ip::Interpolation{2}) = get_n_edges(ip)
get_n_faces(ip::Interpolation{3}) = get_n_surfaces(ip)

get_facelist(ip::Interpolation{1}) = getvertexlist(ip)
get_facelist(ip::Interpolation{2}) = getedgelist(ip)
get_facelist(ip::Interpolation{3}) = getsurfacelist(ip)

# Dim 1 and 2 do not have surfaces
@pure get_n_edges(::Interpolation{1}) = 0
@pure get_n_edges(ip::Interpolation) = length(getedgelist(ip))

@pure get_n_surfaces{order, refshape}(::Interpolation{1, refshape, order}) = 0
@pure get_n_surfaces{order, refshape}(::Interpolation{2, refshape, order}) = 0
@pure get_n_surfaces(ip::Interpolation) = length(getedgelist(ip))

# Dim 1 do not have edges
@pure get_n_vertices(::Interpolation{1, RefCube}) = 2
@pure get_n_vertices(::Interpolation{2, RefCube}) = 4
@pure get_n_vertices(::Interpolation{3, RefCube}) = 8

@pure get_n_vertices(::Interpolation{2, RefTetrahedron}) = 3
@pure get_n_vertices(::Interpolation{3, RefTetrahedron}) = 4

getvertexlist(ip::Interpolation{1, RefCube}) = SVector(SVector(1),
                                                       SVector(2))


# Edges
@pure getedgelist(::Interpolation{2, RefCube}) = SVector(SVector(1, 2),
                                                         SVector(2, 3),
                                                         SVector(3, 4),
                                                         SVector(4, 1))

@pure getedgelist(::Interpolation{2, RefTetrahedron}) = SVector(SVector(1, 2),
                                                             SVector(2, 3),
                                                             SVector(3, 1))
