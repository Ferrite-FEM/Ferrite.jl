.. currentmodule:: JuAFEM

.. _api-solid:


Solid elements
--------------

3 node isoparametric triangle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: plante(ex, ey, ep, D, [eq=[0.0,0.0]]) -> Ke, fe

   .. Docstring generated from Julia source

   Computes the stiffness matrix ``Ke`` and force vector ``fe`` for a three node isoparametric triangular element.

4 node isoparametric quadraterial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: plani4e(ex, ey, ep, D, [eq=[0.0,0.0]]) -> Ke, fe

   .. Docstring generated from Julia source

   Computes the stiffness matrix ``Ke`` and force vector ``fe`` for a four node isoparametric quadraterial element.

8 node isoparametric quadraterial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: plani8e(ex, ey, ep, D, [eq=[0.0,0.0]]) -> Ke, fe

   .. Docstring generated from Julia source

   Computes the stiffness matrix ``Ke`` and force vector ``fe`` for an eight node isoparametric quadraterial element.

8 node isoparametric hexahdron
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: soli8e(ex, ey, ez, ep, D, [eq=[0.0,0.0,0.0]]) -> Ke, fe

   .. Docstring generated from Julia source

   Computes the stiffness matrix ``Ke`` and force vector ``fe`` for a eight node isoparametric hexahedron element.

