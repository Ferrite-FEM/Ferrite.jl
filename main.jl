#! /usr/bin/env julia

using Revise, Ferrite, SparseArrays, ProfileView

# n = 3
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
# function hhhhh(dh)
#     dsp = Ferrite.Final.MallocDSP(ndofs(dh), ndofs(dh))
#     create_sparsity_pattern!(dsp, dh)
#     return dsp
# end
# @time hhhhh(dh);

# dsp = Ferrite.Europe2.MallocDSP4(ndofs(dh), ndofs(dh))

# create_sparsity_pattern!(dsp, dh)



function g(dh)
    dsp = Ferrite.SparsityPattern(ndofs(dh), ndofs(dh))
    create_sparsity_pattern!(dsp, dh)
    return dsp
end

function f(dh)
    dsp = Ferrite.DSP(ndofs(dh), ndofs(dh); rows_per_chunk = 1, growth_factor = 1.5)
    create_sparsity_pattern!(dsp, dh)
    return dsp
end


# mimalloc with n = 40: 21.8% of memory which is sys.maxrss 3.4G
#
#
#
#

function h(dh)
    dsp = Ferrite.MallocDSP(ndofs(dh), ndofs(dh))
    create_sparsity_pattern!(dsp, dh)
    return dsp
end

function hh(dh)
    dsp = Ferrite.MallocDSP3(ndofs(dh), ndofs(dh))
    create_sparsity_pattern!(dsp, dh)
    return dsp
end

function hhh(dh)
    dsp = Ferrite.Europe.MallocDSP4(ndofs(dh), ndofs(dh))
    create_sparsity_pattern!(dsp, dh)
    return dsp
end

function hhhh(dh)
    dsp = Ferrite.Europe2.MallocDSP4(ndofs(dh), ndofs(dh))
    create_sparsity_pattern!(dsp, dh)
    return dsp
end

function hhhhh(dh)
    dsp = Ferrite.Final.MallocDSP(ndofs(dh), ndofs(dh))
    create_sparsity_pattern!(dsp, dh)
    return dsp
end

function hhhhhh(dh)
    dsp = Ferrite.Final.SparsityPattern(ndofs(dh), ndofs(dh))
    # dsp = Ferrite.SparsityPattern(ndofs(dh), ndofs(dh))
    create_sparsity_pattern!(dsp, dh)
    s = 0
    for r in Ferrite.Final.eachrow(dsp)
        for ri in r
            s += ri
        end
    end
    @time finalize(dsp.heap)
    return s
end


function time_me(dh)
    s = hhhhhh(dh)
    return s
end

@time time_me(dh);

function m(dh)
    dsp = Ferrite.MiMallocDSP(ndofs(dh), ndofs(dh); growth_factor = 2)
    create_sparsity_pattern!(dsp, dh);
    # finalize(dsp)
    return dsp
end

function spz(dh)
    dsp = Ferrite.SpzerosDSP(ndofs(dh), ndofs(dh))
    create_sparsity_pattern!(dsp, dh)
    return dsp
end

println("Creating DSP: Vector{Vector{Int}}")
@time g(dh);
GC.gc()
GC.gc()
GC.gc()
@time Ferrite.create_matrix(g(dh));
GC.gc()
GC.gc()
GC.gc()

println("Creating DSP: some chunked garbage version")
@time f(dh);
GC.gc()
GC.gc()
GC.gc()
@time Ferrite.create_matrix(SparseMatrixCSC{Float64, Int}, f(dh));
GC.gc()
GC.gc()
GC.gc()

println("Creating DSP: MemoryBlock thing with free list (basically my own mimalloc?)")
@time h(dh);
GC.gc()
GC.gc()
GC.gc()
@time hh(dh);
GC.gc()
GC.gc()
GC.gc()
@time hhh(dh);
GC.gc()
GC.gc()
GC.gc()
# @profview hhhh(dh);
@time hhhh(dh);
GC.gc()
GC.gc()
GC.gc()
@time hhhhh(dh);
GC.gc()
GC.gc()
GC.gc()
@profview hhhhh(dh)



@time Ferrite.create_matrix(SparseMatrixCSC{Float64, Int}, h(dh));
GC.gc()
GC.gc()
GC.gc()


println("Creating DSP: mimalloc with virtual heap")
@time m(dh);
GC.gc()
GC.gc()
GC.gc()
@time Ferrite.create_matrix(SparseMatrixCSC{Float64, Int}, m(dh));
GC.gc()
GC.gc()
GC.gc()

println("Creating DSP: pushing IJ and spzeros!!")
@time spz(dh);
GC.gc()
GC.gc()
GC.gc()
@time Ferrite.create_matrix(spz(dh));



# Creating DSP: Vector{Vector{Int}}
#   7.127514 seconds (3.29 M allocations: 3.713 GiB, 16.71% gc time)
#   8.506862 seconds (3.29 M allocations: 5.920 GiB, 15.20% gc time)
# Creating DSP: some chunked garbage version
#  12.669635 seconds (4.17 M allocations: 4.702 GiB, 16.35% gc time)
#  14.480421 seconds (4.17 M allocations: 6.909 GiB, 15.83% gc time)
# Creating DSP: MemoryBlock thing with free list (basically my own mimalloc?)
#   6.202821 seconds (568 allocations: 16.480 MiB)
#   7.850059 seconds (576 allocations: 2.223 GiB, 0.09% gc time)
# Creating DSP: mimalloc with virtual heap
#   6.189139 seconds (11 allocations: 16.275 MiB)
#   7.454562 seconds (19 allocations: 2.223 GiB, 0.07% gc time)
# Creating DSP: pusing IJ and spzeros!!
#   9.043348 seconds (59 allocations: 4.082 GiB, 5.64% gc time)
#  12.335072 seconds (67 allocations: 6.790 GiB, 5.61% gc time)
