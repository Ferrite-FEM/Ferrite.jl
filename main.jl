#! /usr/bin/env julia

using Ferrite, SparseArrays, Profile, ProfileView

const n = if length(ARGS) == 1
    parse(Int, ARGS[1])
elseif length(ARGS) == 0
    30
else
    error()
end

const grid = generate_grid(Hexahedron, (n, n, n))
const dh = DofHandler(grid)
add!(dh, :u, Lagrange{RefHexahedron, 2}()^3)
add!(dh, :p, Lagrange{RefHexahedron, 1}())
close!(dh)

function f(dh)
    dsp = Ferrite.DSP(ndofs(dh), ndofs(dh); rows_per_chunk = 1, growth_factor = 1.5)
    create_sparsity_pattern!(dsp, dh)
    return dsp
end

function g(dh)
    dsp = Ferrite.SparsityPattern(ndofs(dh), ndofs(dh))
    create_sparsity_pattern!(dsp, dh)
    return dsp
end

@timev f(dh);
GC.gc()
GC.gc()
GC.gc()
sleep(2)
GC.gc()
GC.gc()
GC.gc()
@profview f(dh);

@timev g(dh);
GC.gc()
GC.gc()
GC.gc()
sleep(2)
GC.gc()
GC.gc()
GC.gc()
@profview g(dh)

# @time dsp = create_sparsity_pattern(dh)
# @time K = create_matrix(dsp)


# begin
#     @time GC.gc()
#     dsp = nothing
# end

# GC.gc()
# GC.gc()
# GC.gc()
# GC.gc()

# x = Profile.take_heap_snapshot()
# @show x
