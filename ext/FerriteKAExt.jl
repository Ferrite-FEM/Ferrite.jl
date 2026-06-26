module FerriteKAExt

using Ferrite, SparseArrays

import Adapt: Adapt, adapt, adapt_structure

import Ferrite: get_grid, AbstractGrid, AbstractDofHandler, get_coordinate_eltype, get_coordinate_type
import Ferrite: get_substruct, distribute_to_tasks
import Ferrite: meandiag, nnodes_per_cell

import KernelAbstractions as KA
import KernelAbstractions: get_backend

import GPUArraysCore: @allowscalar

include("FerriteKAExt/adapt_core.jl")
include("FerriteKAExt/soa_core.jl")

include("FerriteKAExt/device_grid.jl")

include("FerriteKAExt/dof_handler.jl")

include("FerriteKAExt/iterator.jl")

end
