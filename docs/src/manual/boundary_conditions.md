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
