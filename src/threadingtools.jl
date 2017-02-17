module ThreadingTools

using JuAFEM
using Base.Threads

export ThreadVector, @threaded

"""
Wrapper to store things which are supposed to be separate between threads
"""
immutable ThreadVector{T}
    data::Vector{T}
end

@inline Base.getindex(t::ThreadVector, i) = @inbounds return t.data[i]

"""
    @threaded ex

Create a `ThreadVector` with the same length as number of threads
"""
macro threaded(ex)
    _threaded(ex)
end

function _threaded(ex)
    t = nthreads()
    if isa(ex, Symbol) # an identifier
        return esc(quote
            ThreadVector([copy($ex) for i in 1:$t])
        end)
    end
    if isa(ex, Expr) && ex.head == :call
        return esc(quote
            ThreadVector([$ex for i in 1:$t])
        end)
    end
end

# some specialized methods for ThreadVector

"""
    end_assemble(a::ThreadVector{<:Assembler}) -> K

Finalizes a threaded assembly. Returns a sparse matrix with the
assembled values.
"""
function JuAFEM.end_assemble{T <: JuAFEM.Assembler}(a::ThreadVector{T})
    for i in 2:length(a.data) # append all on the first one
        append!(a.data[1].I, a.data[i].I)
        append!(a.data[1].J, a.data[i].J)
        append!(a.data[1].V, a.data[i].V)
    end
    return end_assemble(a.data[1])
end
function JuAFEM.end_assemble{T <: Vector}(a::ThreadVector{T})
    for i in 2:length(a.data) # add all to the first one
        a.data[1] += a.data[i]
    end
    return a.data[1]
end

end # module
