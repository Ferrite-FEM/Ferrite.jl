abstract type AbstractIterator end
abstract type AbstractCellCache end

abstract type AbstractGPUCellCache <: AbstractCellCache end

function makecache(dh::AbstractGPUDofHandler)
    throw(ArgumentError("makecache should be implemented in the derived type"))
end
