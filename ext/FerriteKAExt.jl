module FerriteKAExt

using Ferrite, SparseArrays

import Adapt: Adapt, adapt, adapt_structure
using GPUArraysCore: @allowscalar

import Ferrite: AbstractGrid, AbstractDofHandler

import KernelAbstractions as KA
import KernelAbstractions: get_backend

include("FerriteKAExt/device_grid.jl")
include("FerriteKAExt/dof_handler.jl")
include("FerriteKAExt/adapt_core.jl")
include("FerriteKAExt/soa_core.jl")


end
