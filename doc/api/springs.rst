.. currentmodule:: JuAFEM

.. _api-strings:

********
Springs
********

Basic functions
---------------

.. function:: spring1e(k) -> Matrix{Float64}

   .. Docstring generated from Julia source

   Computes the element stiffness matrix *Ke* for a spring element.

.. function:: spring1s(k, u) -> Vector{Float64}

   .. Docstring generated from Julia source

   Computes the force *fe* for a spring element

