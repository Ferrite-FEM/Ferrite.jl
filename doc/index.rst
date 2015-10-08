Welcome to JuAFEM.jl's documentation!
=====================================


## Differences between CALFEM and JuAFEM.


* CALFEM made some unfortunate choices in how they set up their matrices for dofs and coordinates. Both Julia and MATLAB use [column major order](https://en.wikipedia.org/wiki/Row-major_order) which means that memory is stored column by column. We therefore also want to store for example the dofs for an element in a column.
The concrete effect of this is that the following matrices are transposed in JuAFEM: `Edof`, `Ex`, `Ey`, `Ez`, `Dof`, `Coord`.
The dofs for element `k` is therefore computed by `Edof[2:end, k]` instead of like in MATLAB `Edof(k, 2:end)`.

* Some functions in CALFEM can compute a quantity for many elements at a time. In JuAFEM, you need to loop over the elements and call the function with each separate element.

.. _api:

#####
 API
#####

.. toctree::
    :maxdepth: 2

    api/elements
    api/materials
    api/utilities