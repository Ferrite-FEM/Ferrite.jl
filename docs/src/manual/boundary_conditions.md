```@meta
DocTestSetup = :(using Ferrite)
```

# Boundary Conditions

Every PDE is accompanied with boundary conditions. There are different types of boundary
conditions, and they need to be handled in different ways. Below we discuss how to handle
the most common ones, Dirichlet and Neumann boundary conditions, and how to do it `Ferrite`.

## Dirichlet Boundary Conditions

At a Dirichlet boundary the solution is prescribed to a given value. For the discrete
FE-solution this means that there are some degrees of freedom that are fixed. To be able
to tell which degrees of freedom we should constrain we need the `DofHandler`.

```julia
ch = ConstraintHandler(dh)
```

TBW

!!! note "Examples"
    The following commented examples makes use of Dirichlet boundary conditions:
    - [Heat Equation](@ref)
    - TODO


## Neumann Boundary Conditions

At the Neumann part of the boundary we know something about the gradient of the solution.

As an example, the following code snippet can be included in the element routine,
to evaluate the boundary integral:

```julia
for face in 1:nfaces(cell)
    if onboundary(cell, face) && (cellid(cell), face) ∈ getfaceset(grid, "Neumann Boundary")
        reinit!(facevalues, cell, face)
        for q_point in 1:getnquadpoints(facevalues)
            dΓ = getdetJdV(facevalues, q_point)
            for i in 1:getnbasefunctions(facevalues)
                δu = shape_value(facevalues, q_point, i)
                fe[i] += δu * b * dΓ
            end
        end
    end
end
```

We start by looping over all the faces of the cell, next we have to check if
this particular face is located on the boundary, and then also check that the
face is located on our face-set called `"Neumann Boundary"`. If we have determined
that the current face is indeed on the boundary and in our faceset, then we
reinitialize `facevalues` for this face, using [`reinit!`](@ref). When `reinit!`ing
`facevalues` we also need to give the face number in addition to the cell.
Next we simply loop over the quadrature points of the face, and then loop over
all the test functions and assemble the contribution to the force vector.

!!! note "Examples"
    The following commented examples makes use of Neumann boundary conditions:
    - TODO

## Periodic boundary conditions

Periodic boundary conditions ensure that the solution is periodic. This is common
in multiscale modeling and computational homogenization when solving the finite element
problem on the subscale. The subscale problem is then solved on a Representative Volume
Element (RVE), or Statistical Volume Element (SVE), typically on a cubical domain, to
find the fluctuation field ``u^{\mu}`` with a known macroscopic field ``u^{\mathrm{M}}``
as input. Periodic boundary conditions is one choice for such a setup, and ensures that
the fluctuation is periodic.

A periodic Dirichlet boundary condition is described by

```math
[[u]] = 0 \quad \mathrm{on} \quad \Gamma^{+},
```

where ``[[\bullet]] := \bullet(x^{+}) - \bullet(x^{-})`` is the "jump operator", and ``x^{+}``
and ``x^{-}`` coordinates on the image, ``\Gamma^{+}``, and mirror, ``\Gamma^{-}``, part
of the boundary. Thus, this condition ensure that the jump, or difference, in the solution
between the image and mirror boundary is the zero -- the solution becomes periodic.

In `Ferrite` periodic Dirichlet boundary conditions can be added to the `ConstraintHandler`
by adding a `PeriodicDirichlet` as follows:

```julia
# Create a constraint handler from the dof handler
ch = ConstraintHandler(dofhandler)

# Construct the periodic constraint for field :u
pdbc = PeriodicDirichlet(:u, ["left" => "right", "bottom" => "top"])

# Add the constraint to the constraint handler
add!(ch, pdbc)

# If no more constraints should be added we can close
close!(ch)
```

This adds a constraint on the field `:u` that ensures that the solution on the mirror
boundary face sets (`"left"` and `"bottom"`) mirrors the solution on the image boundary
face sets (`"right"` and `"top"`).

The constraint is imposed in a strong sense, so note
that this requires (i) a periodic domain and (ii) a periodic mesh such that corresponding
element sides can be found on the image and mirror parts of the boundary, respectively.

#### Heterogeneous "periodic" constraint

Instead of solving the just the fluctuation field on the RVE it is possible to solve for
the full field. For example, in the multi-scale setup described above, where the solution
in the RVE is constructed as

```math
u = u^{\mathrm{M}} + u^{\mu},
```

it is possible to solve for ``u`` rather than just ``u^{\mu}``. In order to still ensure
a periodic fluctuation the constraint can instead be described by

```math
[[u]] = [[u^{\mathrm{M}}]] \quad \mathrm{on} \quad \Gamma^{+}.
```

The `PeriodicDirichlet` constraint can be used for this too, although the constraint in this
case is not technically periodic as applied to ``u``, but ensures periodicity on the
fluctuation ``u^{\mu}``. This is done by passing a function to `PeriodicDirichlet`, similar
to `Dirichlet`, which, given the coordinate ``\boldsymbol{x}`` and time `t`, computes the
prescribed values of ``u^{\mathrm{M}}`` on the boundary.

As an example, consider the case where the macroscopic solution is constructed as follows
(first order homogenization):

```math
u^{\mathrm{M}} = \bar{u} + \boldsymbol{\nabla} \bar{u} \cdot [\boldsymbol{x} - \bar{\boldsymbol{x}}],
```

where ``\bar{u}`` is the homogenized field from the macroscale, and ``\bar{\boldsymbol{x}}``
the centre coordinate of the RVE. This corresponds to the following constraint in `Ferrite`:

```julia
# Construct the "periodic" constraint
pdbc = PeriodicDirichlet(
    :u,
    ["left" => "right", "bottom" => "top"],
    (x, t) -> ū + ∇ū ⋅ (x - x̄)
)
```
