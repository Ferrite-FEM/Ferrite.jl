.. currentmodule:: JuAFEM

.. _api-utilities:

*********
Utilities
*********

Plotting
--------

.. function:: eldraw2(ex::AbstractVecOrMat, ey::AbstractVecOrMat, plotpar = [1, 1, 0], elnum::AbstractVector=zeros(0))

   .. Docstring generated from Julia source

   Draws the 2D mesh defined by ex, ey.

.. function:: eldisp2(ex::AbstractVecOrMat, ey::AbstractVecOrMat, ed::AbstractVecOrMat, plotpar = [1, 1, 0], sfac = 1.0)

   .. Docstring generated from Julia source

   Draws the displaced 2D mesh defined by ex, ey and the displacements given in ed

Coordinate extraction
----------------------

.. function:: extract_eldisp(edof::Array{Int}, a::Array{Float64})

   .. Docstring generated from Julia source

   Extracts the element displacements from the global solution vector given an edof matrix. This assumes all elements to have the same number of dofs.

Static condensation
-------------------

.. function:: statcon(K::Matrix, f::Vector, cd::Vector)

   .. Docstring generated from Julia source

   Condenses out the dofs given in cd from K and f.

Assembler
---------

.. function:: start_assemble(N::Int=0)

   .. Docstring generated from Julia source

   Call before starting an assembly.

   Returns an ``Assembler`` type

.. function:: assemble(edof, assembler::Assembler, Ke)

   .. Docstring generated from Julia source

   Assembles the the element stiffness matrix Ke to the global stiffness matrix K.

   Assembles the element matrix ``Ke`` into ``assembler``

.. function:: end_assemble(a::Assembler)

   .. Docstring generated from Julia source

   Finish an assembly. Returns a sparse matrix with the assembled values.

