function task_local end

"""
    task_local(x::T) -> T

Duplicate `x` for a new task such that it can be used concurrently with the original `x`.
This is similar to `copy` but only the data that is known to be mutated, i.e. "scratch
data", are duplicated.

Typically, for concurrent assembly, there are some data structures that can't be shared
between the tasks, for example the local element matrix/vector and the `CellValues`.
`task_local` can thus be used to duplicate theses data structures for each task based on a
"template" data structure. For example,

```julia
# "Template" local matrix and cell values
Ke = zeros(...)
cv = CellValues(...)

# Spawn `ntasks` tasks for concurrent assembly
@sync for i in 1:ntasks
    Threads.@spawn begin
        Ke_task = task_local(Ke)
        cv_task = task_local(Ke)
        for cell in cells_for_task
            assemble_element!(Ke_task, cv_task, ...)
        end
    end
end
```

See the how-to on [multi-threaded assembly](@ref tutorial-threaded-assembly) for a complete
example.

The following "user-facing" types define methods for `task_local`:

 - [`CellValues`](@ref), [`FacetValues`](@ref), [`InterfaceValues`](@ref) are duplicated
   such that they can be `reinit!`ed independently.
 - `DenseArray` (for e.g. the local matrix and vector) are duplicated such that they can be
   modified concurrently.
 - [`CellCache`](@ref) (for caching element nodes and dofs) are duplicated such that they
   can be `reinit!`ed independently.

The following types also define methods for `task_local` but are typically not used directly
by the user but instead used recursively by the above types:

 - [`QuadratureRule`](@ref) and [`FacetQuadratureRule`](@ref)
 - All types which are `isbitstype` (e.g. `Vec`, `Tensor`, `Int`, `Float64`, etc.)
"""
task_local(::Any)

# DenseVector/DenseMatrix (e.g. local matrix and vector)
function task_local(x::T)::T where {S, T <: DenseArray{S}}
    @assert !isbitstype(T)
    if isbitstype(S)
        # If the eltype isbitstype the normal shallow copy can be used...
        return copy(x)::T
    else
        # ... otherwise we recurse and call task_local on the elements
        return map(task_local, x)::T
    end
end

# FacetQuadratureRule can store the QuadratureRules as a tuple
function task_local(x::T)::T where {T <: Tuple}
    if isbitstype(T)
        return x
    else
        return map(task_local, x)::T
    end
end

# General fallback for other types
function task_local(x::T)::T where {T}
    if !isbitstype(T)
        error("MethodError: task_local(::$T) is not implemented")
    end
    return x
end
