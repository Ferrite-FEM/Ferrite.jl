# This files defines the abstract types and interfaces for GPU iterators.
# The concrete implementations are defined in the extension.

# abstract types and interfaces
abstract type AbstractIterator end
abstract type AbstractCellCache end

abstract type AbstractGPUCellCache <: AbstractCellCache end
abstract type AbstractGPUCellIterator <: AbstractIterator end


function _makecache(iterator::AbstractGPUCellIterator, i::Int32)
    throw(ArgumentError("makecache should be implemented in the derived type"))
end

@inline function cellke(::AbstractGPUCellCache)
    throw(ArgumentError("cellke should be implemented in the derived type"))
end

@inline function cellfe(::AbstractGPUCellCache)
    throw(ArgumentError("cellfe should be implemented in the derived type"))
end
