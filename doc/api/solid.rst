.. currentmodule:: JuAFEM

.. _api-solid:


Solid elements
--------------

3 node isoparametric triangle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: plante(ex::VecOrMat, ey::VecOrMat, ep, D::Matrix, eq::VecOrMat=[0.0,0.0])

   .. Docstring generated from Julia source

   Computes the stiffness matrix and force vector for a three node isoparametric triangular element

4 node isoparametric quadraterial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: plani4e(ex::VecOrMat, ey::VecOrMat, ep, D::Matrix, eq::VecOrMat=[0.0,0.0])

   .. Docstring generated from Julia source

   Computes the stiffness matrix and force vector for a four node isoparametric quadraterial element

8 node isoparametric quadraterial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: plani8e(ex::VecOrMat, ey::VecOrMat, ep, D::Matrix, eq::VecOrMat=[0.0,0.0])

   .. Docstring generated from Julia source

   Computes the stiffness matrix and force vector for an eight node isoparametric quadraterial element

8 node isoparametric hexahdron
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: soli8e(ex::VecOrMat, ey::VecOrMat, ep, D::Matrix, eq::VecOrMat=[0.0,0.0,0.0])

   .. Docstring generated from Julia source

   Computes the stiffness matrix and force vector for a eight node isoparametric hexahedron element.

