.. currentmodule:: JuAFEM

.. _api-utilities:

*********
Utilities
*********

Assembler
---------

.. function:: start_assemble(N::Int=0)

   .. Docstring generated from Julia source

   Call before starting an assembly.

   Returns an ``Assembler`` type

.. function:: assemble(edof, assembler::Assembler, Ke)

   .. Docstring generated from Julia source

   Assembles the element matrix ``Ke`` into ``assembler``

.. function:: end_assemble(a::Assembler)

   .. Docstring generated from Julia source

   Finish an assembly. Returns a sparse matrix with the assembled values.

Quadrature
----------

.. function:: start_assemble(N::Int=0)

   .. Docstring generated from Julia source

   Call before starting an assembly.

   Returns an ``Assembler`` type

.. function:: assemble(edof, assembler::Assembler, Ke)

   .. Docstring generated from Julia source

   Assembles the element matrix ``Ke`` into ``assembler``

.. function:: end_assemble(a::Assembler)

   .. Docstring generated from Julia source

   Finish an assembly. Returns a sparse matrix with the assembled values.

