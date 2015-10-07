.. currentmodule:: JuAFEM

.. _api-heat:


Heat elements
-------------

3 node isoparametric triangle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: flw2te(ex, ey, ep, D, [eq=[0.0]]) -> Ke, fe

   .. Docstring generated from Julia source

   Computes the stiffness matrix and ``Ke`` and force vector ``fe`` for a three node isoparametric triangular heat transfer element with a heat source ``eq``\ .

.. function:: flw2ts(ex, ey, ep, D, ed) -> es, et, points

   .. Docstring generated from Julia source

   Computes the heat flows ``es`` and the gradients ``et`` in the gauss points given by the integration rule for a three node isoparametric triangular heat transfer element.

   Also returns the global coordinates ``points`` at the gauss points.

4 node isoparametric quadraterial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: flw2i4e(ex, ey, ep, D, [eq=[0.0]]) -> Ke, fe

   .. Docstring generated from Julia source

   Computes the stiffness matrix ``Ke`` and force vector ``fe`` for a four node isoparametric quadraterial heat transfer element with a heat source ``eq``\ .

.. function:: flw2i4s(ex, ey, ep, D, ed) -> es, et, points

   .. Docstring generated from Julia source

   Computes the heat flows ``es`` and the gradients ``et`` in the gauss points given by the integration rule for a four node isoparametric quadraterial heat transfer element.

   Also returns the global coordinates ``points`` at the gauss points.

8 node isoparametric quadraterial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: flw2i8e(ex, ey, ep, D, [eq=[0.0]]) -> Ke, fe

   .. Docstring generated from Julia source

   Computes the stiffness matrix ``Ke`` and force vector ``fe`` for an eight node isoparametric quadraterial heat transfer element with a heat source ``eq``\ .

.. function:: flw2i8s(ex, ey, ep, D, ed) -> es, et, points

   .. Docstring generated from Julia source

   Computes the heat flows ``es`` and the gradients ``et`` in the gauss points given by the integration rule for an eight node isoparametric quadraterial heat transfer element.

   Also returns the global coordinates ``points`` at the gauss points.

8 node isoparametric hexahdron
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: flw3i8e(ex, ey, ez, ep, D, [eq=[0.0]]) -> Ke, fe

   .. Docstring generated from Julia source

   Computes the stiffness matrix ``Ke`` and force vector ``fe`` for an eight node isoparametric hexahedron heat transfer element with a heat source ``eq``\ .

.. function:: flw3i8s(ex, ey, ez, ep, D, ed) -> es, et, points

   .. Docstring generated from Julia source

   Computes the heat flows ``es`` and the gradients ``et`` in the gauss points given by the integration rule for an eight node isoparametric hexahedron heat transfer element

   Also returns the global coordinates ``points`` at the gauss points.

