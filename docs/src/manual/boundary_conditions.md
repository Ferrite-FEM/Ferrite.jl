```@meta
DocTestSetup = :(using Ferrite)
```

# Initial and Boundary Conditions

Every PDE is accompanied with boundary conditions. There are different types of boundary
conditions, and they need to be handled in different ways. Below we discuss how to handle
the most common ones, Dirichlet and Neumann boundary conditions, and how to do it `Ferrite`.

## Dirichlet Boundary Conditions

At a Dirichlet boundary the unknown field is prescribed to a given value. For the discrete
FE-solution this means that there are some degrees of freedom that are fixed. To handle
Dirichlet boundary conditions in Ferrite we use the [`ConstraintHandler`](@ref). A
constraint handler is created from a DoF handler:

```julia
ch = ConstraintHandler(dh)
```

We can now create Dirichlet constraints and add them to the constraint handler. To create a
Dirichlet constraint we need to specify a field name, a part of the boundary, and a function
for computing the prescribed value. Example:

```julia
dbc1 = Dirichlet(
    :u,                       # Name of the field
    getfaceset(grid, "left"), # Part of the boundary
    x -> 1.0,                 # Function mapping coordinate to a prescribed value
)
```

The field name is given as a symbol, just like when the field was added to the dof handler,
the part of the boundary where this constraint is active is given as a face set, and the
function computing the prescribed value should be of the form `f(x)` or `f(x, t)`
(coordinate `x` and time `t`) and return the prescribed value(s).

!!! note "Multiple sets"
    To apply a constraint on multiple face sets in the grid you can use `union` to join
    them, for example
    ```julia
    left_right = union(getfaceset(grid, "left"), getfaceset(grid, "right"))
    ```
    creates a new face set containing all faces in the `"left"` and "`right`" face sets,
    which can be passed to the `Dirichlet` constructor.

By default the constraint is added to all components of the given field. To add the
constraint to selected components a fourth argument with the components should be passed to
the constructor. Here is an example where a constraint is added to component 1 and 3 of a
vector field `:u`:

```julia
dbc2 = Dirichlet(
    :u,                       # Name of the field
    getfaceset(grid, "left"), # Part of the boundary
    x -> [0.0, 0.0],          # Function mapping coordinate to prescribed values
    [1, 3],                   # Components
)
```

Note that the return value of the function must match with the components -- in the example
above we prescribe components 1 and 3 to 0 so we return a vector of length 2.

Adding the constraints to the constraint handler is done with [`add!`](@ref):

```julia
add!(ch, dbc1)
add!(ch, dbc2)
```

Finally, just like for the dof handler, we need to use [`close!`](@ref) to finalize the
constraint handler. Internally this will then compute the degrees-of-freedom that match the
constraints we added.

If one or more of the constraints depend on time, i.e. they are specified as `f(x, t)`, the
prescribed values can be recomputed in each new time step by calling [`update!`](@ref) with
the proper time, e.g.:

```julia
for t in 0.0:0.1:1.0
    update!(ch, t) # Compute prescribed values for this t
    # Solve for time t...
end
```

!!! note "Examples"
    Most examples make use of Dirichlet boundary conditions, for example [Heat
    Equation](@ref).


## Neumann Boundary Conditions

At the Neumann part of the boundary we know something about the gradient of the solution.

As an example, the following code snippet can be included in the element routine,
to evaluate the boundary integral:

```julia
for face in 1:nfaces(cell)
    if (cellid(cell), face) ∈ getfaceset(grid, "Neumann Boundary")
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

We start by looping over all the faces of the cell, next we check if this particular face is
located on our faceset of interest called `"Neumann Boundary"`. If we have determined
that the current face is indeed on the boundary and in our faceset, then we
reinitialize `facevalues` for this face, using [`reinit!`](@ref). When `reinit!`ing
`facevalues` we also need to give the face number in addition to the cell.
Next we simply loop over the quadrature points of the face, and then loop over
all the test functions and assemble the contribution to the force vector.

!!! note "Examples"
    The following commented examples makes use of Neumann boundary conditions:
    - TODO

## Periodic boundary conditions

Periodic boundary conditions ensure that the solution is periodic across two boundaries. To
define the periodicity we first define the image boundary ``\Gamma^+`` and the mirror
boundary ``\Gamma^-``. We also define a (unique) coordinate mapping between the image and
the mirror: ``\varphi:\ \Gamma^+\, \rightarrow\, \Gamma^-``. With the mapping we can, for
every coordinate on the image, compute the corresponding coordinate on the mirror:

```math
\boldsymbol{x}^- = \varphi(\boldsymbol{x}^+),\quad \boldsymbol{x}^- \in \Gamma^-,\,
\boldsymbol{x}^+ \in \Gamma^+.
```

We now want to ensure that the solution on the image ``\Gamma^+`` is mirrored on the mirror
``\Gamma^-``. This periodicity constraint can thus be described by

```math
u(\boldsymbol{x}^-) = u(\boldsymbol{x}^+).
```

Sometimes this is written as

```math
\llbracket u \rrbracket = 0,
```

where ``\llbracket \bullet \rrbracket := \bullet(\boldsymbol{x}^+) -
\bullet(\boldsymbol{x}^-)`` is the "jump operator". Thus, this condition ensure that the
jump, or difference, in the solution between the image and mirror boundary is the zero --
the solution becomes periodic. For a vector valued problem the periodicity constraint can in
general be written as

```math
\boldsymbol{u}(\boldsymbol{x}^-) = \boldsymbol{R} \cdot \boldsymbol{u}(\boldsymbol{x}^+)
\quad \Leftrightarrow \quad \llbracket \boldsymbol{u} \rrbracket =
\boldsymbol{R} \cdot \boldsymbol{u}(\boldsymbol{x}^+) - \boldsymbol{u}(\boldsymbol{x}^-) =
\boldsymbol{0}
```

where ``\boldsymbol{R}`` is a rotation matrix. If the mapping between mirror and image is
simply a translation (e.g. sides of a cube) this matrix will be the identity matrix.

In `Ferrite` this type of periodic Dirichlet boundary conditions can be added to the
`ConstraintHandler` by constructing an instance of [`PeriodicDirichlet`](@ref). This is
usually done it two steps. First we compute the mapping between mirror and image faces using
[`collect_periodic_faces`](@ref). Here we specify the mirror set and image sets (the sets
are usually known or can be constructed easily ) and the mapping ``\varphi``. Second we
construct the constraint using the `PeriodicDirichlet` constructor. Here we specify which
components of the function that should be constrained, and the rotation matrix
``\boldsymbol{R}`` (when needed). When adding the constraint to the `ConstraintHandler` the
resulting dof-mapping is computed.

Here is a simple example where periodicity is enforced for components 1 and 2 of the field
`:u` between the mirror boundary set `"left"` and the image boundary set `"right"`. Note
that no rotation matrix is needed here since the mirror and image are parallel, just shifted
in the ``x``-direction (as seen by the mapping `φ`):

```julia
# Create a constraint handler from the dof handler
ch = ConstraintHandler(dofhandler)

# Compute the face mapping
φ(x) = x - Vec{2}((1.0, 0.0))
face_mapping = collect_periodic_faces(grid, "left", "right", φ)

# Construct the periodic constraint for field :u
pdbc = PeriodicDirichlet(:u, face_mapping, [1, 2])

# Add the constraint to the constraint handler
add!(ch, pdbc)

# If no more constraints should be added we can close
close!(ch)
```

!!! note
    `PeriodicDirichlet` constraints are imposed in a strong sense, so note that this
    requires a periodic mesh such that it is possible to compute the face mapping between
    faces on the mirror and boundary.

!!! note "Examples"
    Periodic boundary conditions are used in the following examples [Computational
    homogenization](@ref), [Stokes flow](@ref).

#### Heterogeneous "periodic" constraint

It is also possible to define constraints of the form

```math
\llbracket u \rrbracket = \llbracket f \rrbracket
\quad \Leftrightarrow \quad
u(\boldsymbol{x}^+) - u(\boldsymbol{x}^-) =
f(\boldsymbol{x}^+) - f(\boldsymbol{x}^-),
```

where ``f`` is a prescribed function. Although the constraint in this case is not
technically periodic, `PeriodicDirichlet` can be used for this too. This is done by passing
a function to `PeriodicDirichlet`, similar to `Dirichlet`, which, given the coordinate
``\boldsymbol{x}`` and time `t`, computes the prescribed values of ``f`` on the boundary.

Here is an example of how to implement this type of boundary condition, for a known function
`f`:

```julia
pdbc = PeriodicDirichlet(
    :u,
    face_mapping,
    (x, t) -> f(x),
    [1, 2],
)
```

!!! note
    One application for this type of boundary conditions is multiscale modeling and
    computational homogenization when solving the finite element problem for the subscale.
    In this case the unknown ``u`` is split into a macroscopic part ``u^{\mathrm{M}}`` and a
    microscopic/fluctuation part ``u^\mu``, i.e. ``u = u^{\mathrm{M}} + u^{\mu}``.
    Periodicity is then usually enforced for the fluctuation part, i.e. ``\llbracket u^\mu
    \rrbracket = 0``. The equivalent constraint for ``u`` then becomes ``\llbracket u
    \rrbracket = \llbracket u^{\mathrm{M}} \rrbracket``.

    As an example, consider first order homogenization where the macroscopic part is
    constructed as ``u^{\mathrm{M}} = \bar{u} + \boldsymbol{\nabla} \bar{u} \cdot
    [\boldsymbol{x} - \bar{\boldsymbol{x}}]`` for known ``\bar{u}`` and
    ``\boldsymbol{\nabla} \bar{u}``. This could be implemented as
    ```julia
    pdbc = PeriodicDirichlet(
        :u,
        face_mapping,
        (x, t) -> ū + ∇ū  ⋅ (x - x̄)
    )
    ```


## Robin Boundary Conditions
Robin boundary condition, also known as mixed boundary condition, is a type of boundary condition used in mathematical and physical models to describe the behavior of a system at the boundary.In the context of partial differential equations (PDEs), which are equations involving multiple variables and their partial derivatives, a Robin boundary condition combines elements of both Dirichlet and Neumann boundary conditions. From the standard form of KU = F, the additional Robin boundary modifies the element assembly as follows:

```math
\dot{\Phi}(u) + \mathrm{div} \cdot \mathbf{q}(\mathrm{grad} u) = f \quad \text{for all } \mathbf{x} \in \Omega
```

```math
u = g_\mathrm{D} \quad \text{for all } \mathbf{x} \in \Gamma_\mathrm{D}
```

```math
q_\mathrm{n} \equiv \mathbf{q} \cdot \mathbf{n} = g_\mathrm{N} \quad \text{for all } \mathbf{x} \in \Gamma_\mathrm{N}
```

```math
a\,u + b\, q_\mathrm{n} = a\,u + b\, \mathbf{q} \cdot \mathbf{n} = g_\mathrm{R} \quad \text{for all } \mathbf{x} \in \Gamma_\mathrm{R}

```


```julia
for face in 1:nfaces(cell) # Robin BC
        if (cellid(cell), face) ∈ getfaceset(grid, "flux_right")
            reinit!(facevalues, cell, face)
            for q_point in 1:getnquadpoints(facevalues)
                dΓ = getdetJdV(facevalues, q_point)
                for i in 1:n_basefuncs
                    δu = shape_value(facevalues, q_point, i)
                    for j in 1:n_basefuncs
                        u = shape_value(facevalues, q_point, j)
                        Ke[i, j] += K_robin * (δu ⋅ u) * dΓ
                    end
                    # Give the value of q we need. 
                    fe[i] += (δu * K_robin * u_robin) * dΓ # RHS contribution of Robin BC
                end
            end
        end
    end

```

## Initial Conditions

When solving time-dependent problems, initial conditions, different from zero, may be required. 
For finite element formulations of ODE-type, 
i.e. ``\boldsymbol{u}'(t) = \boldsymbol{f}(\boldsymbol{u}(t),t)``, 
where ``\boldsymbol{u}(t)`` are the degrees of freedom,
initial conditions can be specified by the [`apply_analytical!`](@ref) function.
For example, specify the initial pressure as a function of the y-coordinate
```julia
ρ = 1000; g = 9.81    # density [kg/m³] and gravity [N/kg]
grid = generate_grid(Quadrilateral, (10,10))
dh = DofHandler(grid); add!(dh, :u, 2); add!(dh, :p, 1); close!(dh)
u = zeros(ndofs(dh))
apply_analytical!(u, dh, :p, x -> ρ * g * x[2])
```

See also [Time Dependent Problems](@ref) for one example.

!!! note "Consistency"
    `apply_analytical!` does not enforce consistency of the applied solution with the system of 
    equations. Some problems, like for example differential-algebraic systems of equations (DAEs)
    need extra care during initialization. We refer to the paper ["Consistent Initial Condition Calculation for Differential-Algebraic Systems"  by Brown et al.](dx.doi.org/10.1137/S1064827595289996) for more details on this matter.

