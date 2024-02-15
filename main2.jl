using Revise, Ferrite, SparseArrays, ProfileView, BenchmarkTools

const n = if length(ARGS) == 1
    parse(Int, ARGS[1])
elseif length(ARGS) == 0
    20
else
    error()
end

const grid = generate_grid(Hexahedron, (n, n, n))
const dh = DofHandler(grid)
add!(dh, :u, Lagrange{RefHexahedron, 2}()^3)
add!(dh, :p, Lagrange{RefHexahedron, 1}())
close!(dh)

ch = ConstraintHandler(dh)
fm = collect_periodic_faces(grid)
add!(ch, PeriodicDirichlet(:u, fm))
close!(ch)

# function 
# dsp = Ferrite.Final.SparsityPattern(ndofs(dh), ndofs(dh))

# create_sparsity_pattern!(dsp, dh)

function profile_me(dh)
    dsp = Ferrite.Final.SparsityPattern(ndofs(dh), ndofs(dh))
    # dsp = Ferrite.SparsityPattern(ndofs(dh), ndofs(dh))
    create_sparsity_pattern!(dsp, dh)
    condense_sparsity_pattern!(dsp, ch)
    return dsp
end

function bench_me(dh)
    dsp = Ferrite.Final.SparsityPattern(ndofs(dh), ndofs(dh))
    # dsp = Ferrite.SparsityPattern(ndofs(dh), ndofs(dh))
    create_sparsity_pattern!(dsp, dh)
    condense_sparsity_pattern!(dsp, ch)
    s = 0
    for r in Ferrite.eachrow(dsp)
        for ri in r
            s += ri
        end
    end
    finalize(dsp.heap)
    return s
end

function hhhhhh(dh)
    dsp = @showtime Ferrite.Final.SparsityPattern(ndofs(dh), ndofs(dh))
    # dsp = Ferrite.SparsityPattern(ndofs(dh), ndofs(dh))
    @showtime create_sparsity_pattern!(dsp, dh)
    @showtime condense_sparsity_pattern!(dsp, ch)
    s = 0
    for r in Ferrite.eachrow(dsp)
        for ri in r
            s += ri
        end
    end
    @showtime finalize(dsp.heap)
    return s
end

@showtime hhhhhh(dh);
GC.gc()
GC.gc()
GC.gc()
GC.gc()
GC.gc()
GC.gc()
GC.gc()
@showtime hhhhhh(dh);
GC.gc()
GC.gc()
GC.gc()
GC.gc()
GC.gc()
GC.gc()
GC.gc()
display(@benchmark bench_me($dh))

