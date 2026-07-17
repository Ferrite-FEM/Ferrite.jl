using Ferrite
using SparseMatricesCSR: SparseMatrixCSR
using SparseArrays: SparseMatrixCSC
using DataFrames
using Ferrite: FastSparsityPattern

function dh_scalar(grid)
    CT = getcelltype(grid)
    RS = getrefshape(CT)
    return close!(add!(DofHandler(grid), :u, Lagrange{RS, 1}()))
end

function dh_complex(grid::Grid{sdim}) where {sdim}
    CT = getcelltype(grid)
    RS = getrefshape(CT)
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RS, 2}())
    add!(dh, :v, Lagrange{RS, 1}()^sdim)
    return close!(dh)
end

create_sp(dh) = add_sparsity_entries!(init_sparsity_pattern(dh), dh)
create_fsp(dh) = FastSparsityPattern(dh)

grid_2d = generate_grid(Triangle, 1000 .* (1, 1))
grid_3d = generate_grid(Hexahedron, 80 .* (1, 1, 1))

dofhandlers = [
    "dh_2d_scalar" => dh_scalar(grid_2d),
    "dh_3d_scalar" => dh_scalar(grid_3d),
    "dh_2d_complex" => dh_complex(grid_2d),
    "dh_3d_complex" => dh_complex(grid_3d),
]

function timef(f::F, ::Type{MatrixType}, args...; kwargs...) where {F, MatrixType}
    sp0 = f(args...; kwargs...)       # Compile
    allocate_matrix(MatrixType, sp0)  # Compile
    GC.gc()
    sp_allocs = @allocations (sp_runtime = @elapsed (sp = f(args...; kwargs...)))
    m_allocs = @allocations (m_runtime = @elapsed allocate_matrix(MatrixType, sp))
    return (; sp_t = sp_runtime, sp_a = sp_allocs, m_t = m_runtime, m_a = m_allocs)
end

function fmt_time(t::Number; digits = 2)
    units = ["ns", "μs", "ms", "s", "min", "h"]
    values = [1.0e-9, 1.0e-6, 1.0e-3, 1.0, 60.0, 3600]
    idx = findfirst(v -> v > t, values)
    i = max(idx === nothing ? length(values) : idx - 1, 1)
    return string(round(t / values[i]; digits)) * " " * units[i]
end

function fmt_count(n::Integer; digits = 2)
    units = ["k", "M", "G"]
    values = [1.0e3, 1.0e6, 1.0e9]
    n < 1000 && return string(n)
    i = findlast(v -> v ≤ n, values)
    return string(round(n / values[i]; digits)) * units[i]
end

function make_timings(MatrixType)
    return map([create_sp, create_fsp]) do f
        [key => timef(f, MatrixType, dh) for (key, dh) in dofhandlers]
    end
end

function make_df(timings)
    _getdata(v, k) = getindex.(last.(v), k)
    return DataFrame(
        "case" => first.(dofhandlers),
        "t (sp) [s]" => fmt_time.(_getdata(timings[1], :sp_t)),
        "t (fsp) [s]" => fmt_time.(_getdata(timings[2], :sp_t)),
        "allocs (sp)" => fmt_count.(_getdata(timings[1], :sp_a)),
        "allocs (fsp)" => fmt_count.(_getdata(timings[2], :sp_a)),
        "t (K,sp) [s]" => fmt_time.(_getdata(timings[1], :m_t)),
        "t (K,fsp) [s]" => fmt_time.(_getdata(timings[2], :m_t)),
        "allocs (K,sp)" => fmt_count.(_getdata(timings[1], :m_a)),
        "allocs (K,fsp)" => fmt_count.(_getdata(timings[2], :m_a))
    )
end

display(make_df(make_timings(SparseMatrixCSC{Float64, Int})))
# display(make_df(make_timings(SparseMatrixCSR)))
