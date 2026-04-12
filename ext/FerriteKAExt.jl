module FerriteKAExt

using Ferrite, SparseArrays

import Adapt: Adapt, adapt, adapt_structure
using GPUArraysCore: @allowscalar

import Ferrite: get_grid, AbstractGrid, AbstractDofHandler, get_coordinate_eltype
import Ferrite: as_structure_of_arrays, get_substruct
import Ferrite: meandiag, nnodes_per_cell
import Ferrite: CellCache, ImmutableCellCache

import KernelAbstractions as KA
import KernelAbstractions: get_backend

include("FerriteKAExt/adapt_core.jl")
include("FerriteKAExt/device_grid.jl")
include("FerriteKAExt/dof_handler.jl")

include("FerriteKAExt/soa_core.jl")



include("FerriteKAExt/iterator.jl")

end
