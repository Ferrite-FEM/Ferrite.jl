.. currentmodule:: JuAFEM

.. _api-materials:

********
Materials
********

Basic functions
---------------

.. function:: hooke(ptype, E, v)

   .. Docstring generated from Julia source

   Computes the material stiffness matrix for a linear elastic and isotropic material.

   **ptype**:

     * 1: plane stress

     * 2: plane strain

     * 3: axisymmetry

     * 4: 3D

