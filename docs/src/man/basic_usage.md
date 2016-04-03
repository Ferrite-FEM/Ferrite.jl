# Basic Usage

Create a quadrature rule:

```julia
julia> quad_rule = QuadratureRule(:legendre, Dim{2}, RefCube(), 2);
```

Create a function space

```julia
julia> func_space = Lagrange{2, RefCube, 1}()
```

Use these to create a `FEValues` object.

```julia
julia> fe_values = FEValues(quad_rule, func_space);
```

Use `reinit!` to update `fe_values` on an element:

```julia
julia> x = Vec{2, Float64}[Vec{2}((0.0, 0.0)),
                           Vec{2}((1.5, 0.0)),
                           Vec{2}((2.0, 2.0)),
                           Vec{2}((0.0, 1.0))];

julia> reinit!(fe_values, x);
```

We can now query the `FEValues` object for shape function information:

Values of shape functions in quadrature point 3

```julia
julia> shape_value(fe_values, 3)
4-element Array{Float64,1}:
 0.166667
 0.0446582
 0.166667
 0.622008
```

Derivatives of shape function 2 in quadrature point 1

```julia
 julia> shape_gradient(fe_values, 1, 2)
2-element ContMechTensors.Tensor{1,2,Float64,2}:
  0.520116
 -0.219827

```

We can also evaluate values and gradients of functions on the finite element basis:

```julia
julia> T = [0.0, 1.0, 2.0, 1.5];

julia> function_scalar_value(fe_values, 3, T) # value of T in 3rd quad point
1.311004233964073

julia> function_scalar_gradient(fe_values, 1, T)  # value of grad T in 1st quad point
2-element ContMechTensors.Tensor{1,2,Float64,2}:
 0.410202
 1.1153
```

For a vector valued function:

```julia
julia> u = Vec{2, Float64}[Vec{2}((0.0, 0.0)),
                          Vec{2}((3.5, 2.0)),
                          Vec{2}((2.0, 2.0)),
                          Vec{2}((2.0, 1.0))];

julia> function_vector_value(fe_values, 2, u)
2-element ContMechTensors.Tensor{1,2,Float64,2}:
 2.59968
 1.62201

julia> function_vector_symmetric_gradient(fe_values, 3, u)
2x2 ContMechTensors.SymmetricTensor{2,2,Float64,3}:
 -0.0443518  0.713306
  0.713306   0.617741
```

For more functions see  the documentation for [`FEValues`](../lib/maintypes#JuAFEM.FEValues)