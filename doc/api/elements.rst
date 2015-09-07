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

Bars
----

.. function:: bar2e(ex::Array{Float64}, ey::Array{Float64}, elem_prop::Array{Float64})

   .. Docstring generated from Julia source

   Computes the element stiffness matrix *Ke* for a 2D bar element.

.. function:: bar2s(ex::Array{Float64}, ey::Array{Float64}, elem_prop::Array{Float64}, el_disp::Array{Float64})

   .. Docstring generated from Julia source

   Computes the sectional force (normal force) *N* for a 2D bar element.

4 node isoparametric quadraterial
---------------------------------

.. function:: plani4e(ex::Vector, ey::Vector, ep, D, eq=[0.0,0.0])

   .. Docstring generated from Julia source

   Computes the stiffness matrix for a four node isoparametric quadraterial element

