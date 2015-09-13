.. currentmodule:: JuAFEM

.. _api-heat:


Heat elements
-------------

3 node isoparametric triangle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: flw2te(ex, ey, ep, D, [eq=[0.0]]) -> Ke, fe

   .. Docstring generated from Julia source

   Computes the stiffness matrix and ``Ke`` and force vector ``fe`` for a three node isoparametric triangular heat transfer element.

4 node isoparametric quadraterial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: flw2i4e(ex, ey, ep, D, [eq=[0.0]]) -> Ke, fe

   .. Docstring generated from Julia source

   Computes the stiffness matrix ``Ke`` and force vector ``fe`` for a four node isoparametric quadraterial heat transfer element.

8 node isoparametric quadraterial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: flw2i8e(ex, ey, ep, D, [eq=[0.0]]) -> Ke, fe

   .. Docstring generated from Julia source

   Computes the stiffness matrix ``Ke`` and force vector ``fe`` for an eight node isoparametric quadraterial element.

8 node isoparametric hexahdron
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: flw3i8e(ex, ey, ez, ep, D, [eq=[0.0]]) -> Ke, fe

   .. Docstring generated from Julia source

   Computes the stiffness matrix ``Ke`` and force vector ``fe`` for a eight node isoparametric hexahedron heat transfer  element.

