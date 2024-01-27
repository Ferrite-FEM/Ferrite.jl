
mutable struct SpzerosDSP <: AbstractSparsityPattern
    const nrows::Int
    const ncols::Int
    const I::Vector{Int}
    const J::Vector{Int}
    function SpzerosDSP(
            nrows::Int, ncols::Int;
            nnz_per_row::Int = 8,
            growth_factor::Number = 2,
        )
        return new(nrows, ncols, Int[], Int[])
    end
end

n_rows(dsp::SpzerosDSP) = dsp.nrows
n_cols(dsp::SpzerosDSP) = dsp.ncols

function add_entry!(dsp::SpzerosDSP, row::Int, col::Int)
    @boundscheck (1 <= row <= n_rows(dsp) && 1 <= col <= n_cols(dsp)) || throw(BoundsError())
    push!(dsp.I, row)
    push!(dsp.J, col)
    return
end

function create_matrix(dsp::SpzerosDSP)
    return spzeros!!(Float64, dsp.I, dsp.J, dsp.nrows, dsp.ncols)
end
