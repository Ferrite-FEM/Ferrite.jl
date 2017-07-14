```@meta
DocTestSetup = quote
    using JuAFEM
    quad_rule = QuadratureRule{2, RefCube}(2)
    interpolation = Lagrange{2, RefCube, 1}()
    cell_values = CellScalarValues(quad_rule, interpolation)
    x = Vec{2, Float64}[Vec{2}((0.0, 0.0)),
                           Vec{2}((1.5, 0.0)),
                           Vec{2}((2.0, 2.0)),
                           Vec{2}((0.0, 1.0))]
    reinit!(cell_values, x)
end
```

# Getting started

!!! tip
    Checkout some examples of usage of JuAFEM in the
    [`examples/`](https://github.com/KristofferC/JuAFEM.jl/tree/master/examples) directory.

For the impatient: Here is a quick overview on how the some of the packages
functionalities can be used. This quickly describes [`CellScalarValues`](@ref CellValues)
which a lot of the package is built upon.

First, create a quadrature rule, for integration in 2D, on a reference cube:

```jldoctest
julia> quad_rule = QuadratureRule{2, RefCube}(2);
```

Next, create an interpolation

```jldoctest
julia> interpolation = Lagrange{2, RefCube, 1}();
```

Use these to create a `CellScalarValues` object.

```jldoctest
julia> cell_values = CellScalarValues(quad_rule, interpolation);
```

Presume one cell in the grid has the following vertices:

```jldoctest
julia> x = Vec{2, Float64}[Vec{2}((0.0, 0.0)),
                           Vec{2}((1.5, 0.0)),
                           Vec{2}((2.0, 2.0)),
                           Vec{2}((0.0, 1.0))];
```

To update `cell_values` for the given cell, use `reinit!`:

```jldoctest
julia> reinit!(cell_values, x)
```

We can now query the `CellScalarValues` object for shape function information:

Value of shape function 1 in quadrature point 3

```jldoctest
julia> shape_value(cell_values, 3, 1)
0.16666666666666669
```

Derivative of the same shape function, in the same quadrature point

```jldoctest
julia> shape_gradient(cell_values, 3, 1)
2-element Tensors.Tensor{1,2,Float64,2}:
  0.165523
 -0.665523
```

We can also evaluate values and gradients of functions on the finite element basis.

```jldoctest
julia> T = [0.0, 1.0, 2.0, 1.5]; # nodal values

julia> function_value(cell_values, 3, T) # value of T in 3rd quad point
1.3110042339640733

julia> function_gradient(cell_values, 1, T)  # value of grad(T) in 1st quad point
2-element Tensors.Tensor{1,2,Float64,2}:
 0.410202
 1.1153
```

The same can also be done for a vector valued function:

```jldoctest
julia> u = Vec{2, Float64}[Vec{2}((0.0, 0.0)),
                           Vec{2}((3.5, 2.0)),
                           Vec{2}((2.0, 2.0)),
                           Vec{2}((2.0, 1.0))]; # nodal vectors

julia> function_value(cell_values, 2, u) # value of u in 2nd quad point
2-element Tensors.Tensor{1,2,Float64,2}:
 2.59968
 1.62201

julia> function_symmetric_gradient(cell_values, 3, u) # sym(grad(u)) in 3rd quad point
2×2 Tensors.SymmetricTensor{2,2,Float64,3}:
 -0.0443518  0.713306
  0.713306   0.617741
```

For more functions see the documentation for [`CellValues`](@ref).
