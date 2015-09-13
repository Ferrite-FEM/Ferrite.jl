.. currentmodule:: JuAFEM

.. _api-bars:

Bars
----

.. function:: bar2e(ex, ey, elem_prop) -> Ke

   .. Docstring generated from Julia source

   Computes the element stiffness matrix ``Ke`` for a 2D bar element.

.. function:: bar2s(ex, ey, elem_prop, el_disp) -> N

   .. Docstring generated from Julia source

   Computes the sectional force (normal force) ``N`` for a 2D bar element.

.. function:: bar2g(ex, ey, elem_prop, N) -> Ke

   .. Docstring generated from Julia source

   Computes the element stiffness matrix ``Ke`` for a geometrically nonlinear 2D bar element.

