# [Interpolations](@id devdocs-interpolations)

All `Interpolation`s should subtype `Interpolation{shape, order}`,
where `shape <: AbstractRefShape` is the reference shape for which
the interpolation is defined and [`order`](@ref Ferrite.getorder) is the characteristic interpolation
order. The [how-to at bottom of this page](@ref devdocs-howto_new-interpolation) describes how to implement a new interpolation.

## Methods to be implemented for a new interpolation
### Always
```@docs
Ferrite.reference_shape_value(::Interpolation, ::Vec, ::Int)
Ferrite.reference_coordinates(::Interpolation)
Ferrite.vertexdof_indices(::Interpolation)
Ferrite.facedof_indices(::Interpolation)
Ferrite.facedof_interior_indices(::Interpolation)
Ferrite.edgedof_indices(::Interpolation)
Ferrite.edgedof_interior_indices(::Interpolation)
Ferrite.volumedof_interior_indices(::Interpolation)
Ferrite.getnbasefunctions(::Interpolation)
Ferrite.adjust_dofs_during_distribution(::Interpolation)
```

### For special interpolations
#### Discontinuous interpolations
For discontinuous interpolations, implementing the following methods might be required to apply Dirichlet boundary conditions.
```@docs
Ferrite.is_discontinuous(::Interpolation)
Ferrite.dirichlet_vertexdof_indices(::Interpolation)
Ferrite.dirichlet_facedof_indices(::Interpolation)
Ferrite.dirichlet_edgedof_indices(::Interpolation)
```

#### Non-identity mapping
For interpolations that have a non-identity mapping (see
[Mapping of finite elements](@ref mapping_theory)), the
mapping type must be specified.
```@docs
Ferrite.mapping_type
Ferrite.get_direction
```

### The defaults should always work
The following functions are defined such that they should work for
any interpolations that defines the required functions specified above.
```@docs
Ferrite.getrefshape(::Interpolation)
Ferrite.getorder(::Interpolation)
Ferrite.reference_shape_gradient(::Interpolation, ::Vec, ::Int)
Ferrite.reference_shape_gradient_and_value(::Interpolation, ::Vec, ::Int)
Ferrite.reference_shape_hessian_gradient_and_value(::Interpolation, ::Vec, ::Int)
Ferrite.boundarydof_indices
Ferrite.dirichlet_boundarydof_indices
Ferrite.reference_shape_values!
Ferrite.reference_shape_gradients!
Ferrite.reference_shape_gradients_and_values!
Ferrite.reference_shape_hessians_gradients_and_values!
Ferrite.shape_value_type(ip::Interpolation, ::Type{T}) where T<:Number
```

## [How to implement a new interpolation](@id devdocs-howto_new-interpolation)
!!! warning
    The API for implementing a new interpolation is not fully stable yet.

As an example, we will implement a `Q`uadratic `T`riangle `I`nterpolation
(`QTI`), corresponding to `Lagrange{RefTriangle, 2}()`. Since this interpolation
gives scalar-valued shape functions, it is a subtype of `ScalarInterpolation`,

```@example InterpolationExample
using Ferrite # hide
using Test #hide
struct QTI <: ScalarInterpolation{RefTriangle, 2}
end
ip_qti = QTI() # hide
ip_lag = Lagrange{RefTriangle, 2}() # hide
nothing # hide
```
Before we jump in and define the shape functions associated with this
interpolation, we first need to consider the `RefTriangle` entities,

```@docs; canonical=false
RefTriangle
```

For this particular interpolation, we have one degree of freedom associated
with each vertex, and one degree of freedom associated with each edge.
Following the [Ferrite numbering rules](@ref "Ordering-of-dofs"), we start by enumerating the
vertices first, followed by the edges. The numbering is based on the `RefTriangle` shown above, and the actual shape functions are taken from [defelement.org](https://defelement.org/elements/examples/triangle-lagrange-equispaced-2.html).
```@example InterpolationExample
function Ferrite.reference_shape_value(ip::QTI, ξ::Vec{2}, shape_number::Int)
    ξ₁ = ξ[1]
    ξ₂ = ξ[2]
    γ = 1 - ξ₁ - ξ₂ # Helper
    shape_number == 1 && return ξ₁ * (2ξ₁ - 1) # v1: 1 at ξ = (1, 0)
    shape_number == 2 && return ξ₂ * (2ξ₂ - 1) # v2: 1 at ξ = (0, 1)
    shape_number == 3 && return γ * (2γ - 1)   # v3: 1 at ξ = (0, 0)
    shape_number == 4 && return 4ξ₁ * ξ₂ # e1: 1 at ξ = (½, ½)
    shape_number == 5 && return 4ξ₂ * γ  # e2: 1 at ξ = (0, ½)
    shape_number == 6 && return 4ξ₁ * γ  # e3: 1 at ξ = (½, 0)
    throw(ArgumentError("no shape function $shape_number for interpolation $ip"))
end
function compare_test(f::Function, op = isequal) # hide
    @test op(f(ip_lag), f(ip_qti)) #TODO: Change to @assert # hide
    return nothing #hide
end # hide
ξ = rand(Vec{2}) / 2 # hide
Ns = foreach(i -> compare_test(ip -> Ferrite.reference_shape_value(ip, ξ, i), isapprox), 1:6) # hide
```
Having defined the actual interpolation function, we must now provide the
information in comments above to Ferrite. We start by providing the reference
coordinate for each shape number,
```@example InterpolationExample
function Ferrite.reference_coordinates(::QTI)
    return [
        Vec{2, Float64}((1.0, 0.0)), # v1
        Vec{2, Float64}((0.0, 1.0)), # v2
        Vec{2, Float64}((0.0, 0.0)), # v3
        Vec{2, Float64}((0.5, 0.5)), # center of e1
        Vec{2, Float64}((0.0, 0.5)), # center of e2
        Vec{2, Float64}((0.5, 0.0)), # center of e3
    ]
end
compare_test(Ferrite.reference_coordinates, isapprox) # hide
```
We move on to defining which dof indices belong to each vertex.
As we have 3 vertices on the `RefTriangle`, this should be a tuple
with length 3, and each element contains all dof indices for the particular
vertex. In this case, we only have a single dof per vertex,
```@example InterpolationExample
Ferrite.vertexdof_indices(::QTI) = ((1,), (2,), (3,))
compare_test(Ferrite.vertexdof_indices) # hide
```
Note that the dofs are assigned in order of increasing codimension, followed by the 
index of the local geometry, consistent with its local orientation. E.g. first all
dofs of the first vertex, then all dofs of the second vertex, and so on. The dof index
can be arbitrarily assigned, as long as they are assigned consistently and between 1 and
the number of basis functions (here 6).
```@example InterpolationExample
#                                           e1    e2    e3
Ferrite.edgedof_interior_indices(::QTI) = ((4,), (5,), (6,))
compare_test(Ferrite.edgedof_interior_indices) # hide
#                                  v1 v2 e1    v2 v3 e2    v3 v1 e3
Ferrite.edgedof_indices(::QTI) = ((1, 2, 4,), (2, 3, 5,), (3, 1, 6,))
compare_test(Ferrite.edgedof_indices) # hide
```
But here we need two functions, one for the `interior` indices (those that
have not yet been included in lower-dimensional entities (vertices in this
case)), and one for all indices for dofs that belong to the edge.

For the triangle, we only have a single face. However, all the dofs that
belong to the face, also belongs to either the vertices or edges,
hence we have no "interior" face dofs. So we get,
```@example InterpolationExample
Ferrite.facedof_interior_indices(::QTI) = ((),)
compare_test(Ferrite.facedof_interior_indices) # hide
Ferrite.facedof_indices(::QTI) = ((1, 2, 3, 4, 5, 6),)
compare_test(Ferrite.facedof_indices) # hide
```

Finally, since this is a 2d element, we have no `volumedofs`, and thus
```@example InterpolationExample
Ferrite.volumedof_interior_indices(::QTI) = ()
compare_test(Ferrite.volumedof_interior_indices)            # hide
```

It is necessary to tell Ferrite the total number of base functions, e.g.,
```@example InterpolationExample
Ferrite.getnbasefunctions(::QTI) = 6
```

For distributing the degrees of freedom, higher order interpolations
require that we account for the ordering on their entity. For example,
if we have two interior dofs associated with an edge, we must match
them the edges of the for the cells that share the edge, to make sure
we consider the same ordering. Since we only have a single interior
dof per edge, we don't need to adjust these, hence,
```@example InterpolationExample
Ferrite.adjust_dofs_during_distribution(::QTI) = false
```

!!! tip
    The function `test_interpolation_properties` in `test/test_interpolations.jl`
    can be used when implementation to check that some basic properties are fullfilled.

```@example InterpolationExample
grid = generate_grid(Triangle, (2,2))                       # hide
dh_qti = close!(add!(DofHandler(grid), :u, ip_qti))         # hide
dh_lag = close!(add!(DofHandler(grid), :u, ip_lag))         # hide
@test ndofs(dh_qti) == ndofs(dh_lag)                        # hide
@test dof_range(dh_qti, :u) == dof_range(dh_lag, :u)        # hide
@test celldofs(dh_qti, 1) == celldofs(dh_lag, 1)            # hide
qr = QuadratureRule{RefTriangle}(2)                         # hide
ipg = Lagrange{RefTriangle, 1}()                            # hide
x = Vec.([(0.0, 0.0), (0.5, 0.5), (0.1, 1.0)])              # hide
cv_qti = CellValues(qr, ip_qti, ipg)                        # hide
cv_lag = CellValues(qr, ip_lag, ipg)                        # hide
reinit!.((cv_qti, cv_lag), (x,))                            # hide
N_qti, N_lag = shape_value.((cv_qti, cv_lag), 1, 1)         # hide
@test N_qti ≈ N_lag                                         # hide
dNdx_qti, dNdx_lag = shape_gradient.((cv_qti, cv_lag), 1, 1)# hide
@test dNdx_qti ≈ dNdx_lag                                   # hide
nothing                                                     # hide
```
