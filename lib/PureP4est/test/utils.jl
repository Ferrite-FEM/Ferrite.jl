# Coarse meshes as tree vertex tuples (global vertex ids in counter-clockwise/VTK local
# order), numbered exactly like Ferrite's `generate_grid` and the `generate_simple_disc_grid`
# of Ferrite's test utilities, so tree ids and expectations carry over between the suites.

"""
    brick(dims::NTuple{2, Int}) -> Vector{NTuple{4, Int}}
    brick(dims::NTuple{3, Int}) -> Vector{NTuple{8, Int}}

Structured `nx × ny (× nz)` brick of quadrilateral/hexahedral trees: vertices numbered
x-fastest, then y, then z; trees x-fastest, like `generate_grid(Quadrilateral/Hexahedron, dims)`.
"""
function brick((nx, ny)::NTuple{2, Int})
    quad(n1) = (n1, n1 + 1, n1 + nx + 2, n1 + nx + 1)
    return [quad((j - 1) * (nx + 1) + i) for j in 1:ny for i in 1:nx]
end
function brick((nx, ny, nz)::NTuple{3, Int})
    nlayer = (nx + 1) * (ny + 1)
    quad(n1) = (n1, n1 + 1, n1 + nx + 2, n1 + nx + 1)
    hex(n1) = (quad(n1)..., (quad(n1) .+ nlayer)...)
    return [hex((k - 1) * nlayer + (j - 1) * (nx + 1) + i) for k in 1:nz for j in 1:ny for i in 1:nx]
end

"""
    brick_position(dims, k) -> NTuple{dim, Int}

Zero-based integer position of tree `k` of `brick(dims)`: the tree covers the unit box
`position .+ [0, 1]^dim` (for the unpermuted brick, whose trees are axis-aligned).
"""
brick_position(dims::NTuple{dim, Int}, k::Integer) where {dim} = Tuple(CartesianIndices(dims)[k]) .- 1

"""
    disc(n) -> Vector{NTuple{4, Int}}
    disc3(n) -> Vector{NTuple{8, Int}}

`n` kite-shaped quadrilaterals around a central vertex (`2n + 1`), and the single-layer
hexahedral extrusion thereof — the topology of `generate_simple_disc_grid(Quadrilateral/
Hexahedron, n)`.
"""
function disc(n::Int)
    nnodes = 2n + 1
    return [(2i - 1, 2i, 2i + 1 == nnodes ? 1 : 2i + 1, nnodes) for i in 1:n]
end
disc3(n::Int) = [(q..., (q .+ (2n + 1))...) for q in disc(n)]

"""
    nhanging(ln::PureP4est.LNodes) -> Int

Number of distinct hanging nodes recorded by `lnodes`.
"""
nhanging(ln) = length(unique!(vcat(first.(ln.hanging2), first.(ln.hanging4))))

"""
    nnodes(ln::PureP4est.LNodes) -> Int

Number of nodes numbered by `lnodes`.
"""
nnodes(ln) = length(ln.noderefs)
