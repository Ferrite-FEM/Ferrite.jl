.. currentmodule:: JuAFEM

.. _api-materials:

*********
Materials
*********

Linear elastic
--------------

.. function:: hooke(ptype, E, v) -> D

   .. Docstring generated from Julia source

   Computes the material stiffness matrix for a linear elastic and isotropic material with elastic modulus ``E`` and poissons ratio ``v``\ .

   **ptype**:

   * 1: plane stress

   * 2: plane strain

   * 3: axisymmetry

   * 4: 3D

