.. currentmodule:: JuAFEM

.. _api-elements:

********
Elements
********

Springs
-------

.. function:: spring1e(k) -> Matrix{Float64}

   .. Docstring generated from Julia source

   Computes the element stiffness matrix *Ke* for a spring element.

.. function:: spring1s(k, u) -> Vector{Float64}

   .. Docstring generated from Julia source

   Computes the force *fe* for a spring element

4 node isoparametric quadraterial
---------------------------------

.. function:: plani4e(ex::Vector, ey::Vector, ep, D, eq=[0.0,0.0])

   .. Docstring generated from Julia source

   Computes the stiffness matrix for a four node isoparametric quadraterial element

