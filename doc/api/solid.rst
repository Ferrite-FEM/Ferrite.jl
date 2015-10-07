.. currentmodule:: JuAFEM

.. _api-solid:


Solid elements
--------------

3 node isoparametric triangle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: plante(ex, ey, ep, D, [eq = zeros(2)]) -> Ke, fe

   .. Docstring generated from Julia source

   Computes the stiffness matrix ``Ke`` and force vector ``fe`` for a three node isoparametric triangular element with body load ``eq``\ .

.. function:: plants(ex, ey, ep, D, ed) -> σs, εs, points

   .. Docstring generated from Julia source

   Computes the stresses and strains in the gauss points given by the integration rule for a three node isoparametric triangular element.

   Also returns the global coordinates ``points`` at the gauss points.

.. function:: plantf(ex, ey, ep, σ) -> intf

   .. Docstring generated from Julia source

   Computes the internal forces from ``σ`` for a three node isoparametric triangular element.

4 node isoparametric quadraterial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: plani4e(ex, ey, ep, D, [eq = zeros(2)]) -> Ke, fe

   .. Docstring generated from Julia source

   Computes the stiffness matrix ``Ke`` and force vector ``fe`` for a four node isoparametric quadraterial element with body load ``eq``\ .

.. function:: plani4s(ex, ey, ep, D, ed) -> σs, εs, points

   .. Docstring generated from Julia source

   Computes the stresses and strains in the gauss points given by the integration rule for a four node isoparametric quadraterial element.

   Also returns the global coordinates ``points`` at the gauss points.

.. function:: plani4f(ex, ey, ep, σ) -> intf

   .. Docstring generated from Julia source

   Computes the internal forces from ``σ`` for a four node isoparametric quadraterial element.

8 node isoparametric quadraterial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: plani8e(ex, ey, ep, D, [eq = zeros(2)]) -> Ke, fe

   .. Docstring generated from Julia source

   Computes the stiffness matrix ``Ke`` and force vector ``fe`` for an eight node isoparametric quadraterial element with body load ``eq``\ .

.. function:: plani8s(ex, ey, ep, D, ed) -> σs, εs, points

   .. Docstring generated from Julia source

   Computes the stresses and strains in the gauss points given by the integration rule for an eight node isoparametric quadraterial element.

   Also returns the global coordinates ``points`` at the gauss points.

.. function:: plani8f(ex, ey, ep, σ) -> intf

   .. Docstring generated from Julia source

   Computes the internal forces from ``σ`` for  an eight node isoparametric quadraterial element.

8 node isoparametric hexahedron
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: soli8e(ex, ey, ez, ep, D, [eq = zeros(3)]) -> Ke, fe

   .. Docstring generated from Julia source

   Computes the stiffness matrix ``Ke`` and force vector ``fe`` for an eight node isoparametric hexahedron element with body load ``eq``\ .

.. function:: soli8s(ex, ey, ez, ep, D, ed) -> σs, εs, points

   .. Docstring generated from Julia source

   Computes the stresses and strains in the gauss points given by the integration rule for an eight node isoparametric hexahedron element.

   Also returns the global coordinates ``points`` at the gauss points.

.. function:: soli8f(ex, ey, ep, σ) -> intf

   .. Docstring generated from Julia source

   Computes the internal forces from ``σ`` for an eight node isoparametric hexahedron element.

