function task_local end

"""
    task_local(A)

Duplicate `A` for a new task.
"""
task_local(::Any)

# Vector/Matrix (e.g. local matrix and vector)
task_local(A::Array) = copy(A)

# To help with struct fields which are Union{X, Nothing}
task_local(::Nothing) = nothing
