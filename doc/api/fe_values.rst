.. currentmodule:: JuAFEM

.. _api-fe_values:

FE Values
---------

To assist with creating new finite elements ``JuAFEM`` has the concept of an object called
``FEValues``. The idea for this was taken from deal.ii_.

An ``FEValues`` object facilitates with evaluating shape functions, gradient of shape functions and evaluating values on operators of finite element discretized functions.

Initializing an ``FEValues`` object is done with two other objects. One ``FunctionSpace`` object and one ``QuadratureRule``.

Function spaces
^^^^^^^^^^^^^^^
A function space is described by its name, for example ``Lagrange``, an order of the base functions and a shape which the space is defined on.

The following function spaces are currently available:

- ``Lagrange{1, JuaFEM.Line}``
- ``Lagrange{2, JuaFEM.Line}``
- ``Lagrange{1, JuaFEM.Square}``
- ``Lagrange{1, JuaFEM.Triangle}``
- ``Lagrange{1, JuaFEM.Triangle}``
- ``Lagrange{2, JuaFEM.Triangle}``
- ``Lagrange{1, JuaFEM.Cube}``
- ``Serendipity{2, JuaFEM.Square}``

Quadrature
^^^^^^^^^^

Currently only Gauss quadrature is implemented. A `QuadratureRule` is created from ``get_gaussrule(shape, order)`` where shape is an instance of one of the shapes shown in the function space list above and order is the order of the quadrature rule.


Using FE Values
^^^^^^^^^^^^^^^

An example of creating a ``FEValues`` object is shown below.

.. code-block:: julia
    quad_rule = get_gaussrule(JuAFEM.Square(), 2)
    func_space = func_space = Lagrange{1, JuaFEM.Square}()
    fe_values = FEValues(func_space, quad_rule)

Upon creation, ``FEValues`` caches the values of the shape functions and derivatives in the quadrature points.

The points of FEValues is that for each element you call ``reinit!(fev, x)`` where ``x`` is a `Vector` of `Tensor{1}`s. This will update the global shape function derivatives, jacobians, weights etc.

Different queries can now be performed.

Shape function queries
^^^^^^^^^^^^^^^^^^^^^^

.. function:: shape_value(fe_v, q_point::Int) -> value

   .. Docstring generated from Julia source

   Gets the value of the shape function for a given quadrature point

.. function:: shape_value(fe_v, q_point::Int, base_func::Int) -> value

   .. Docstring generated from Julia source

   Gets the value of the shape function at a given quadrature point and given base function

.. function:: shape_gradient(fe_v, q_point::Int) -> gradients::Vector{Tensor{2}}

   .. Docstring generated from Julia source

   Get the gradients of the shape functions for a given quadrature point

.. function:: shape_gradient(fe_v, q_point::Int, base_func::Int) -> gradient::Tensor{2}

   .. Docstring generated from Julia source

   Get the gradient of the shape functions for a given quadrature point and base function

Discretized function queries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can also compute different operatiors on a finite element discretized function:

.. function:: function_scalar_value(fe_v, q_point::Int, u::Vector) -> value

   .. Docstring generated from Julia source

   Computes the value in a quadrature point for a scalar valued function

.. function:: function_vector_value(fe_v, q_point::Int, u::Vector{Tensor{1}}) -> value::Tensor{1}

   .. Docstring generated from Julia source

   Computes the value in a quadrature point for a vector valued function.

.. function:: function_scalar_gradient(fe_v, q_point::Int, u::Vector) -> grad::Tensor{1}

   .. Docstring generated from Julia source

   Computes the gradient in a quadrature point for a scalar valued function.

.. function:: function_vector_symmetric_gradient(fe_v, q_point::Int, u::Vector{Tensor{1}}) -> sym_grad::SymmetricTensor{2}

   .. Docstring generated from Julia source

   Computes the symmetric gradient (jacobian) in a quadrature point for a vector valued function. Result is stored in ``grad``\ .

.. function:: function_vector_divergence(fe_v, q_point::Int, u::Vector{Tensor{1}}) -> divergence

   .. Docstring generated from Julia source

   Computes the divergence in a quadrature point for a vector valued function.

.. _https://www.dealii.org/developer/doxygen/deal.II/classFEValues.html
