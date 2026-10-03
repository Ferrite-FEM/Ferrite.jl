function task_local_copy end

"""
    task_local_copy(x::T) -> T

Duplicate `x` for a new task such that it can be used concurrently with the original `x`.
This is similar to `copy` but only the data that is known to be mutated, i.e. "scratch
data", are duplicated.

Typically, for concurrent assembly, there are some data structures that can't be shared
between the tasks, for example the local element matrix/vector and the `CellValues`.
`task_local_copy` can thus be used to duplicate these data structures for each task based
on a "template" data structure. For example,

```julia
# "Template" local matrix and cell values
Ke = zeros(...)
cv = CellValues(...)

# Spawn `ntasks` tasks for concurrent assembly
@sync for i in 1:ntasks
    Threads.@spawn begin
        Ke_task = task_local_copy(Ke)
        cv_task = task_local_copy(cv)
        for cell in cells_for_task
            assemble_element!(Ke_task, cv_task, ...)
        end
    end
end
```

See the how-to on [multi-threaded assembly](@ref howto-threaded-assembly) for a complete
example.

The following "user-facing" types define methods for `task_local_copy`:

 - [`CellValues`](@ref), [`MultiFieldCellValues`](@ref), [`FacetValues`](@ref),
   [`InterfaceValues`](@ref), and [`PointValues`](@ref) are duplicated such that they can
   be `reinit!`ed independently.
 - `DenseArray` (for e.g. the local matrix and vector) are duplicated such that they can be
   modified concurrently.
 - [`CellCache`](@ref), [`FacetCache`](@ref), and [`InterfaceCache`](@ref) (for caching
   element nodes and dofs) are duplicated such that they can be `reinit!`ed independently.
 - Assemblers returned by [`start_assemble`](@ref) are duplicated such that they can be
   used concurrently: the internal buffers are duplicated but the global matrix and vector
   are shared. Note that concurrent assembly still requires that tasks don't write to the
   same entries at the same time, e.g. by using grid coloring, or by using an atomic
   assembler (`start_assemble(...; atomic = true)`).

The following types also define methods for `task_local_copy` but are typically not used
directly by the user but instead used recursively by the above types:

 - [`QuadratureRule`](@ref) and [`FacetQuadratureRule`](@ref)
 - All types which are `isbitstype` (e.g. `Vec`, `Tensor`, `Int`, `Float64`, etc.)
"""
task_local_copy(::Any)

# DenseVector/DenseMatrix (e.g. local matrix and vector)
function task_local_copy(x::T)::T where {S, T <: DenseArray{S}}
    @assert !isbitstype(T)
    if isbitstype(S)
        # If the eltype isbitstype the normal shallow copy can be used...
        return copy(x)::T
    else
        # ... otherwise we recurse and call task_local_copy on the elements
        return map(task_local_copy, x)::T
    end
end

# FacetQuadratureRule can store the QuadratureRules as a tuple
function task_local_copy(x::T)::T where {T <: Tuple}
    if isbitstype(T)
        return x
    else
        return map(task_local_copy, x)::T
    end
end

# General fallback for other types
function task_local_copy(x::T)::T where {T}
    if !isbitstype(T)
        throw(MethodError(task_local_copy, (x,)))
    end
    return x
end
