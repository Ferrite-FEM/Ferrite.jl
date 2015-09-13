.. currentmodule:: JuAFEM

.. _api-utilities:

*********
Utilities
*********

Plotting
--------

.. function:: eldraw2(ex, ey, [plotpar=[1,1,0], elnum=zeros(0)])

   .. Docstring generated from Julia source

   Draws the 2D mesh defined by ``ex``\ , ``ey``\ .

.. function:: eldisp2(ex, ey, ed, [plotpar=[1,1,0], sfac=1.0])

   .. Docstring generated from Julia source

   Draws the displaced 2D mesh defined by ``ex``\ , ``ey`` and the displacements given in ``ed``\ .

Solving system of equations
----------------------------

.. function:: solveq(K, f, bc, [symmetric=false]) -> a, fb

   .. Docstring generated from Julia source

   Solves the equation system Ka = f taking into account the Dirichlet boundary conditions in the matrix ``bc``\ . Returns the solution vector ``a`` and reaction forces ``fb`` If ``symmetric`` is set to ``true``\ , the matrix will be factorized with Cholesky factorization.

Coordinate extraction
----------------------

.. function:: extract(edof, a)

   .. Docstring generated from Julia source

   Extracts the element displacements from the global solution vector ``a`` given an ``edof`` matrix. This assumes all elements to have the same number of dofs.

Static condensation
-------------------

.. function:: statcon(K, f, cd) -> K_cond, f_cond

   .. Docstring generated from Julia source

   Condenses out the dofs given in cd from K and f.

Assembler
---------

.. function:: start_assemble([N=0]) -> Assembler

   .. Docstring generated from Julia source

   Call before starting an assembly.

   Returns an ``Assembler`` type that is used to hold the intermediate data before an assembly is finished.

.. function:: assemble(edof, a, Ke)

   .. Docstring generated from Julia source

   Assembles the element matrix ``Ke`` into ``a``\ .

.. function:: end_assemble(a::Assembler) -> K

   .. Docstring generated from Julia source

   Finalizes an assembly. Returns a sparse matrix with the assembled values.

