.. currentmodule:: JuAFEM

.. _api-utilities:

*********
Utilities
*********


Export to VTK
-------------

.. function:: vtk_grid(Edof, Coord, Dof, nen, filename::AbstractString) -> vtkgrid

   .. Docstring generated from Julia source

   Creates an unstructured VTK grid. ``nen`` is the number of nodes per element

   To add cell data and point data and write the file see https://github.com/jipolanco/WriteVTK.jl#generating-an-unstructured-vtk-file


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

