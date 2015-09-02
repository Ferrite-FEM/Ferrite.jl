.. currentmodule:: JuAFEM

.. _api-quadrature:

**********
Quadrature
**********



Basic functions
---------------

.. function:: make_quadrule(order::Int) -> Float

   .. Docstring generated from Julia source

   Creates a ``GaussQuadratureRule`` that integrates functions on a square to the given order.

.. function:: integrate(qr::GaussQuadratureRule, f)

   .. Docstring generated from Julia source

   Integrates the function *f* with the given ``GaussQuadratureRule``

