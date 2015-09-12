.. currentmodule:: JuAFEM

.. _api-elements:

********
Elements
********

Springs
-------

.. function:: spring1e(k) -> Matrix

   .. Docstring generated from Julia source

   Computes the element stiffness matrix *Ke* for a spring element.

.. function:: spring1s(k, u) -> Vector

   .. Docstring generated from Julia source

   Computes the force *fe* for a spring element

Bars
----

.. function:: bar2e(ex::VecOrMat, ey::VecOrMat, elem_prop::VecOrMat)

   .. Docstring generated from Julia source

   Computes the element stiffness matrix *Ke* for a 2D bar element.

.. function:: bar2s(ex::VecOrMat, ey::VecOrMat, elem_prop::VecOrMat, el_disp::VecOrMat)

   .. Docstring generated from Julia source

   Computes the sectional force (normal force) *N* for a 2D bar element.

3 node isoparametric triangle
------------------------------

.. function:: plante(ex::VecOrMat, ey::VecOrMat, ep, D::Matrix, eq::VecOrMat=[0.0,0.0])

   .. Docstring generated from Julia source

   Computes the stiffness matrix for a three node isoparametric triangular element

4 node isoparametric quadraterial
---------------------------------

.. function:: plani4e(ex::VecOrMat, ey::VecOrMat, ep, D::Matrix, eq::VecOrMat=[0.0,0.0])

   .. Docstring generated from Julia source

   Computes the stiffness matrix for a four node isoparametric quadraterial element

8 node isoparametric quadraterial
---------------------------------
.. function:: plani8e(ex::VecOrMat, ey::VecOrMat, ep, D::Matrix, eq::VecOrMat=[0.0,0.0])

   .. Docstring generated from Julia source

   Computes the stiffness matrix for a eight node isoparametric quadraterial element

8 node isoparametric hexahdron
---------------------------------
.. function:: soli8e(ex::VecOrMat, ey::VecOrMat, ep, D::Matrix, eq::VecOrMat=[0.0,0.0])

   .. Docstring generated from Julia source

   Computes the stiffness matrix for a 8node isoparametric hexaedric element

