```@meta
DocTestSetup = :(using Ferrite)
```

# [Assembly](@id man-assembly)

When the local stiffness matrix and force vector have been calculated
they should be assembled into the global stiffness matrix and the
global force vector. This is just a matter of adding the local
matrix and vector to the global one, at the correct place. Consider e.g.
assembling the local stiffness matrix `ke` and the local force vector `fe`
into the global `K` and `f` respectively. These should be assembled into
the row/column which corresponds to the degrees of freedom for the cell:

```julia
K[celldofs, celldofs] += ke
f[celldofs]           += fe
```

where `celldofs` is the vector containing the degrees of freedom for the cell.
The method above is very inefficient -- it is especially costly to index
into the sparse matrix `K` directly (see [Comparison of assembly strategies](@ref)
for details). Therefore we will instead use an
`Assembler` that will help with the assembling of both the global stiffness
matrix and the global force vector. It is also often convenient to create the
sparse matrix just once, and reuse the allocated matrix. This is useful for
e.g. iterative solvers or time dependent problems where the sparse matrix
structure, or [Sparsity Pattern](@ref "Sparsity pattern and sparse matrices")
will stay the same in every iteration/time step.

## `Assembler`

Assembling efficiently into the sparse matrix requires some extra workspace.
This workspace is allocated in an `Assembler`. [`start_assemble`](@ref) is
used to create an `Assembler`:

```julia
A = start_assemble(K)
A = start_assemble(K, f)
```

where `K` is the global stiffness matrix, and `f` the global force vector.
It is optional to pass the force vector to the assembler -- sometimes
there is no need to assemble a global force vector.

The [`assemble!`](@ref) function is used to assemble element contributions
to the assembler. For example, to assemble the element tangent stiffness `ke`
and the element force vector `fe` to the assembler `A`, the following code can
be used:

```julia
assemble!(A, celldofs, ke)
assemble!(A, celldofs, ke, fe)
```

which perform the following operations in an efficient manner:

```julia
K[celldofs, celldofs] += ke
f[celldofs]           += fe
```

## Pseudo-code for efficient assembly

Quite often the same sparsity pattern can be reused multiple times. For example:

 - For time-dependent problems the pattern can be reused for all timesteps
 - For non-linear problems the pattern can be reused for all iterations

In such cases it is enough to construct the global matrix `K` once. Below is
some pseudo-code for how to do this for a time-dependent problem:

```julia
K = allocate_matrix(dh)
f = zeros(ndofs(dh))

for t in 1:timesteps
    A = start_assemble(K, f) # start_assemble zeroes K and f
    for cell in CellIterator(dh)
        ke, fe = element_routine(...)
        assemble!(A, celldofs(cell), ke, fe)
    end
    # Apply boundary conditions and solve for u(t)
    # ...
end
```

## Comparison of assembly strategies

As discussed above there are various ways to assemble the local matrix into the global one.
In particular, it was mentioned that naive indexing is very inefficient and that using an
assembler is faster. To put some concrete numbers to these statements we will compare some
strategies in this section. First we compare just a single assembly operation (e.g.
assembling an already computed local matrix) and then to relate this to a more realistic
scenario we compare the full matrix assembly including the integration of all the elements.

!!! note "Pre-allocated global matrix"
    All strategies that we compare below uses a pre-allocated global matrix `K` with the
    correct sparsity pattern. Starting with something like
    `K = spzeros(ndofs(dh), ndofs(dh))` and then inserting entries is excruciatingly slow
    due to the sparse data structure so this method is not even considered.

For the comparison we need a representative global matrix to assemble into. In the following
setup code we create grid with triangles and a DofHandler with a quadratic scalar field.
From this we instantiate the global matrix.

```@example assembly-perf
using Ferrite

# Quadratic scalar interpolation
ip = Lagrange{RefTriangle, 2}()

# DofHandler
const N = 100
grid = generate_grid(Triangle, (N, N))
const dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)

# Global matrix and a corresponding assembler
const K = allocate_matrix(dh)
nothing # hide
```

#### Strategy 1: matrix indexing

The first strategy is to index directly, using the vector of global dofs, into the global
matrix:

```@example assembly-perf
function assemble_v1(_, K, dofs, Ke)
    K[dofs, dofs] += Ke
    return
end
nothing # hide
```

This looks very simple but it is very inefficient (as the numbers will show later). To
understand why the operation `K[dofs, dofs] += Ke` (with `K` being a sparse matrix) is so
slow we can dig into the details.

In Julia there is no "`+=`"-operation and so `x += y` is identical to `x = x + y`.
Translating this to our example we have

```julia
K[dofs, dofs] = K[dofs, dofs] + Ke
```

We can break down this a bit further into these equivalent three steps:

```julia
tmp1 = K[dofs, dofs]   # 1
tmp2 = tmp1 + Ke       # 2
K[dofs, dofs] = tmp2   # 3
```

Now the problem with this strategy becomes a bit more obvious:
 - In line 1 there is first an allocation of a new matrix (`tmp1`) followed by indexing into
   `K` to copy elements from `K` to `tmp1`. Both of these operations are rather costly:
   allocations should always be minimized in tight loops, and indexing into a sparse matrix
   is non-trivial due to the data structure. In addition, since the `dofs` vector contains
   the global indices (which are not sorted nor consecutive) we have a random access
   pattern.
 - In line 2 there is another allocation of a matrix (`tmp2`) for the result of the addition
   of `tmp1` and `Ke`.
 - In line 3 we again need to index into the sparse matrix to copy over the elements from
   `tmp2` to `K`. This essentially duplicates the indexing effort from line 1 since we need
   to lookup the same locations in `K` again.

!!! note "Broadcasting"
    Using [broadcasting](https://docs.julialang.org/en/v1/manual/arrays/#Broadcasting), e.g.
    `K[dofs, dofs] .+= Ke` is an alternative to the above, and resembles a `+=`-operation.
    In theory this should be as efficient as the explicit loop presented in the next
    section.

#### Strategy 2: scalar indexing

A variant of the first strategy is to explicitly loop over the indices and add the elements
individually as scalars:

```@example assembly-perf
function assemble_v2(_, K, dofs, Ke)
    for (i, I) in pairs(dofs)
        for (j, J) in pairs(dofs)
            K[I, J] += Ke[i, j]
        end
    end
    return
end
nothing # hide
```

The core operation, `K[I, J] += Ke[i, j]`, can still be broken down into three equivalent
steps:

```julia
tmp1 = K[I, J]
tmp2 = tmp1 + Ke[i, j]
K[I, J] = tmp2
```

The key difference here is that we index using integers (`I`, `J`, `i`, and `j`) which means
that `tmp1` and `tmp2` are scalars which don't need to be allocated on the heap. This
stragety thus eliminates all allocations that were present in the first strategy. However,
we still lookup the same location in `K` twice, and we still have a random access pattern.

#### Strategy 3: scalar indexing with single lookup

To improve on the second strategy we will get rid of the double lookup into the sparse
matrix `K`. While Julia doesn't have a "`+=`"-operation there is a `addindex!`-function in
Ferrite which does exactly what we want: it adds a value to a specific location in a sparse
matrix using a single lookup.

```@example assembly-perf
function assemble_v3(_, K, dofs, Ke)
    for (i, I) in pairs(dofs)
        for (j, J) in pairs(dofs)
            Ferrite.addindex!(K, Ke[i, j], I, J)
        end
    end
    return
end
nothing # hide
```

With this method we remove the double lookup, but the issue of random access patterns still
remains.

#### Strategy 4: using an assembler

Finally, the last strategy we consider uses an assembler. The assembler is a specific
datastructure that pre-allocates some workspace to make the assembly more efficient:

```@example assembly-perf
function assemble_v4(assembler, _, dofs, Ke)
    assemble!(assembler, dofs, Ke)
    return
end
nothing # hide
```

The extra workspace inside the assembler is used to sort the dofs when `assemble!` is
called. After sorting it is possible to loop over the sparse matrix data structure and
insert all elements of `Ke` in one go instead of having to lookup locations randomly.

### Single element assembly

First we will compare the four functions above for a single assembly operation, i.e.
inserting one local matrix into the global matrix. For this we simply create a random local
matrix since we are not conserned with the actual values. We also pick the "middle" element
and extract the dofs for that element. Finally, an assembler is created with
`start_assemble` to use with the fourth strategy.

```@example assembly-perf
dofs_per_cell = ndofs_per_cell(dh)
const Ke = rand(dofs_per_cell, dofs_per_cell)
const dofs = celldofs(dh, N * N ÷ 2)

const assembler = start_assemble(K)
nothing # hide
```

We use BenchmarkTools to measure the performance:

```@example assembly-perf
assemble_v1(assembler, K, dofs, Ke) # hide
assemble_v2(assembler, K, dofs, Ke) # hide
assemble_v3(assembler, K, dofs, Ke) # hide
assemble_v4(assembler, K, dofs, Ke) # hide
nothing                             # hide
```

```julia
using BenchmarkTools

@btime assemble_v1(assembler, K, dofs, Ke) evals = 10 setup = Ferrite.fillzero!(K)
@btime assemble_v2(assembler, K, dofs, Ke) evals = 10 setup = Ferrite.fillzero!(K)
@btime assemble_v3(assembler, K, dofs, Ke) evals = 10 setup = Ferrite.fillzero!(K)
@btime assemble_v4(assembler, K, dofs, Ke) evals = 10 setup = Ferrite.fillzero!(K)
```

The results below are obtained on an Macbook Pro with an Apple M3 CPU.

```
606.438 μs (36 allocations: 7.67 MiB)
283.300 ns (0 allocations: 0 bytes)
158.300 ns (0 allocations: 0 bytes)
 83.400 ns (0 allocations: 0 bytes)
```

The results match what we expect based on the explanations above:
 - Between strategy 1 and 2 we got rid of the allocations completely and decreased the time
   with a factor of 2100(!).
 - Between strategy 2 and 3 we got rid of the double lookup and decreased the time with
   another factor of almost 2.
 - Between strategy 3 and 4 we got rid of the random lookup order and decreased the time
   with another factor of almost 2.

The most important thing for this benchmark is to get rid of the allocations. By using an
assembler instead of doing the naive thing we reduce the runtime with a factor of more than
7000(!!) in total.

### Full system assembly

We will now compare the four strategies in a more realistic scenario where we assemble all
elements. This is to put the assembly performance in relation to other operations in the
finite element program. After all, assembly performance might not matter in the end if other
things dominate the runtime anyway.

For this comparison we simply consider the heat equation (see
[Tutorial 1: Heat equation](../tutorials/heat_equation.md)) and assemble the global matrix.

```@example assembly-perf
function assemble_system!(assembler_function::F, K, dh, cv) where {F}
    assembler = start_assemble(K)
    ke = zeros(ndofs_per_cell(dh), ndofs_per_cell(dh))
    n = getnbasefunctions(cv)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        ke .= 0
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            for i in 1:n
                ∇ϕi = shape_gradient(cv, qp, i)
                for j in 1:n
                    ∇ϕj = shape_gradient(cv, qp, j)
                    ke[i, j] += ( ∇ϕi ⋅ ∇ϕj ) * dΩ
                end
            end
        end
        assembler_function(assembler, K, celldofs(cell), ke)
    end
    return
end
nothing # hide
```

Finally, we need cellvalues for the field in order to perform the integration:

```@example assembly-perf
qr = QuadratureRule{RefTriangle}(2)
const cellvalues = CellValues(qr, ip)
nothing # hide
```

We can now time the four assembly strategies:

```@example assembly-perf
res = 1138.8803468514259                           # hide
# assemble_system!(assemble_v1, K, dh, cellvalues) # hide
# @assert norm(K.nzval) ≈ res                      # hide
assemble_system!(assemble_v2, K, dh, cellvalues)   # hide
@assert norm(K.nzval) ≈ res                        # hide
assemble_system!(assemble_v3, K, dh, cellvalues)   # hide
@assert norm(K.nzval) ≈ res                        # hide
assemble_system!(assemble_v4, K, dh, cellvalues)   # hide
@assert norm(K.nzval) ≈ res                        # hide
nothing                                            # hide
```

```julia
@time assemble_system!(assemble_v1, K, dh, cellvalues)
@time assemble_system!(assemble_v2, K, dh, cellvalues)
@time assemble_system!(assemble_v3, K, dh, cellvalues)
@time assemble_system!(assemble_v4, K, dh, cellvalues)
```

We then obtain the following results (running on the same machine as above):

```
12.175625 seconds (719.99 k allocations: 149.809 GiB, 11.59% gc time)
 0.009313 seconds (8 allocations: 928 bytes)
 0.006055 seconds (8 allocations: 928 bytes)
 0.004530 seconds (10 allocations: 1.062 KiB)
```

This follows the same trend as for the benchmarks for individual cell assembly and shows
that the efficiency of the assembly strategy is crucial for the overall performance of the
program. In particular this benchmark shows that allocations in such a tight loop from the
first strategy is very costly and puts a strain on the garbage collector: 11% of the time is
spent in GC instead of crunching numbers.

It should of course be noted that the more expensive the element routine is, the less the
performance of the assembly strategy matters for the total runtime. However, there are no
reason not to use the fastest method given that it is readily available in Ferrite.
