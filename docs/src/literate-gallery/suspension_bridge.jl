# # [Suspension bridge under a moving train](@id tutorial-suspension-bridge)
#
# ![](suspension_bridge-light.webp)
# ![](suspension_bridge-dark.webp)
#
# *Figure 1*: Axial stress in the bridge members as the train crosses (displacements
# exaggerated 40×).
#
# In the spirit of bridge builder games, this example constructs a suspension bridge
# from pin-jointed bars and drives a train across it, following the axial stress in
# every member. It showcases:
#
# * bar (truss) elements: `Line` cells *embedded* in 2D, where the displacement has
#   more components than the reference dimension of the interpolation,
# * building a `Grid` manually, node by node and bar by bar,
# * reusing a factorization when solving many load cases with the same matrix,
# * writing a time series where one part of the scene (the train) changes geometry.
#
# ## Bar elements
#
# Each member is a straight two-node bar that only carries axial force. With the unit
# tangent $\boldsymbol{t}$ along the bar, the axial strain and stress are
# ```math
# \epsilon = \boldsymbol{t} \cdot \boldsymbol{\nabla u} \cdot \boldsymbol{t}, \qquad
# \sigma = E \epsilon,
# ```
# and with cross section area $A$ the equilibrium equation in weak form reads: find
# $\boldsymbol{u}$ such that
# ```math
# \sum_{\text{bars}} \int E A \,
# (\boldsymbol{t} \cdot \boldsymbol{\nabla \delta u} \cdot \boldsymbol{t})
# (\boldsymbol{t} \cdot \boldsymbol{\nabla u} \cdot \boldsymbol{t}) \,
# \mathrm{d}s
# = \sum_{\text{bars}} \int \boldsymbol{\delta u} \cdot \boldsymbol{b} \, \mathrm{d}s
# \quad \forall \, \boldsymbol{\delta u},
# ```
# where $\boldsymbol{b}$ is the distributed load per meter: the weight of the member
# itself and, on the roadway bars, the weight of the train above them.
#
# No special truss element is needed for this: we use ordinary `Line` cells whose
# nodes have 2D coordinates, and a vector-valued interpolation `Lagrange{RefLine}^2`.
# For such *embedded* elements Ferrite computes gradients with respect to the spatial
# coordinates through the pseudo-inverse of the (non-square) Jacobian, so
# `shape_gradient` and `function_gradient` work just like for regular elements, and
# the integration measure from `getdetJdV` is the arc length.
#
# Everything is linear: small displacements, and the cables are ordinary bars with no
# slack or tension-only behavior. Good enough for a game — don't build a real bridge
# from this program.
#
# ## Building the bridge
#
# The bridge deck is a [Warren truss](https://en.wikipedia.org/wiki/Truss_bridge)
# girder that spans between two abutments, with two towers standing on piers. The main
# cables run from anchorages at the abutments over the tower tops, sagging as a
# parabola between them, and vertical hangers carry the deck from the cables.
using Ferrite
using LinearAlgebra, SparseArrays
using VTKHDF, WriteVTK

L = 60.0            # span between the abutments
a = 2.0             # deck panel length
h = 2.0             # height of the deck girder
H = 12.0            # tower height above the deck
sag = 7.0           # main cable sag
x_towers = (12.0, 48.0);

# The grid is assembled by hand from `Node`s and `Line` cells. Small helper functions
# push a node/bar and return/record its index; each bar is also tagged with the member
# group it belongs to, which later determines its cross section and weight.
nodes = Node{2, Float64}[]
cells = Line[]
groups = Dict(name => Int[] for name in ("roadway", "girder", "cables", "hangers", "towers"))
addnode!(x, y) = (push!(nodes, Node(Vec(x, y))); length(nodes))
addbar!(group, n1, n2) = (push!(cells, Line((n1, n2))); push!(groups[group], length(cells)); nothing)

# First the deck girder: roadway nodes on top, lower chord nodes staggered half a
# panel, and diagonals zigzagging between them so that the girder is fully
# triangulated (a chain of pin-jointed bars would otherwise be a mechanism):
deck = [addnode!(x, 0.0) for x in 0:a:L]
chord = [addnode!(x, -h) for x in (a / 2):a:(L - a / 2)]
for i in 1:(length(deck) - 1)
    addbar!("roadway", deck[i], deck[i + 1])
    addbar!("girder", deck[i], chord[i])
    addbar!("girder", chord[i], deck[i + 1])
    i < length(chord) && addbar!("girder", chord[i], chord[i + 1])
end

# Then the towers and the cable system. The main cable hangs as a parabola between the
# tower tops with a hanger down to the roadway at every panel point, and the backstays
# anchor the tower tops at the abutments:
deckindex(x) = round(Int, x / a) + 1
towertops = [addnode!(x, H) for x in x_towers]
addbar!("towers", deck[deckindex(x_towers[1])], towertops[1])
addbar!("towers", deck[deckindex(x_towers[2])], towertops[2])

x1, x2 = x_towers
prev = towertops[1]
for x in (x1 + a):a:(x2 - a)
    ξ = (x - x1) / (x2 - x1)
    n = addnode!(x, H - 4 * sag * ξ * (1 - ξ))
    addbar!("cables", prev, n)
    addbar!("hangers", n, deck[deckindex(x)])
    global prev = n
end
addbar!("cables", prev, towertops[2])
addbar!("cables", deck[1], towertops[1])
addbar!("cables", towertops[2], deck[end])

grid = Grid(cells, nodes);

# All members are steel, with cross section areas per group, collected into per-cell
# vectors. The mass per meter `μ` (used for the dead load) also includes the roadway
# itself — asphalt, sleepers and rails weigh more than the bars that carry them:
E = 210.0e9         # Young's modulus [Pa]
ρ = 7850.0          # density [kg/m³]
g = 9.81            # gravity [m/s²]
areas = Dict(
    "roadway" => 8.0e-3, "girder" => 8.0e-3, "cables" => 8.0e-3,
    "hangers" => 1.2e-3, "towers" => 5.0e-2,
)
A = zeros(getncells(grid))
for (name, cellids) in groups
    A[cellids] .= areas[name]
end
μ = ρ .* A
μ[groups["roadway"]] .+= 2000.0;

# ## FE preliminaries
#
# The displacement field is vector-valued on the one-dimensional reference line —
# this is what makes the elements embedded. The same interpolation describes the
# geometry, and since strain (and thus everything in the integrand) is constant along
# a linear bar, a single quadrature point suffices:
ip = Lagrange{RefLine, 1}()
cv = CellValues(QuadratureRule{RefLine}(1), ip^2, ip^2)

dh = DofHandler(grid)
add!(dh, :u, ip^2)
close!(dh);

# The bridge is pinned at the abutments and at the two tower piers:
ch = ConstraintHandler(dh)
supports = Set([deck[1], deck[end], deck[deckindex(x1)], deck[deckindex(x2)]])
add!(ch, Dirichlet(:u, supports, Returns(zero(Vec{2}))))
close!(ch);

# Assembly of the stiffness matrix and the dead load follows the standard pattern.
# The tangent is computed from the cell coordinates, and note how `getdetJdV` provides
# the length measure $\mathrm{d}s$:
function assemble_system!(K, f, dh, cv, A, μ)
    n = getnbasefunctions(cv)
    ke = zeros(n, n)
    fe = zeros(n)
    assembler = start_assemble(K, f)
    for cc in CellIterator(dh)
        reinit!(cv, cc)
        x = getcoordinates(cc)
        t = (x[2] - x[1]) / norm(x[2] - x[1])
        EA = E * A[cellid(cc)]
        w = Vec(0.0, -g * μ[cellid(cc)]) # weight per meter
        fill!(ke, 0.0)
        fill!(fe, 0.0)
        for qp in 1:getnquadpoints(cv)
            ds = getdetJdV(cv, qp)
            for i in 1:n
                fe[i] += (shape_value(cv, qp, i) ⋅ w) * ds
                Bi = t ⋅ shape_gradient(cv, qp, i) ⋅ t
                for j in 1:n
                    Bj = t ⋅ shape_gradient(cv, qp, j) ⋅ t
                    ke[i, j] += EA * Bi * Bj * ds
                end
            end
        end
        assemble!(assembler, celldofs(cc), ke, fe)
    end
    return
end

K = allocate_matrix(dh)
f_dead = zeros(ndofs(dh))
assemble_system!(K, f_dead, dh, cv, A, μ);

# Every load case uses the same stiffness matrix, so we condense the constraints into
# `K` once, factorize it once, and only update the right hand side in the load loop
# (see [`get_rhs_data`](@ref) and [`apply_rhs!`](@ref)):
rhsdata = get_rhs_data(ch, K)
apply!(K, ch)
Kfact = cholesky(Symmetric(K));

# After each solve the member stress is recovered from the axial strain
# $\boldsymbol{t} \cdot \boldsymbol{\nabla u} \cdot \boldsymbol{t}$ at the (single)
# quadrature point:
function bar_stresses!(σ, dh, cv, u)
    for cc in CellIterator(dh)
        reinit!(cv, cc)
        x = getcoordinates(cc)
        t = (x[2] - x[1]) / norm(x[2] - x[1])
        ∇u = function_gradient(cv, 1, u[celldofs(cc)])
        σ[cellid(cc)] = E * (t ⋅ ∇u ⋅ t)
    end
    return σ
end;

# ## The train
#
# The train is a locomotive and two wagons, described by the offset of each car behind
# the head of the train, its length, mass, and height (the height only matters for
# drawing). Each car spreads its weight uniformly over its footprint, so together they
# define a moving line load $q(x)$ on the roadway:
train = [
    (front = 0.0, length = 9.0, mass = 100.0e3, height = 3.4),  # locomotive
    (front = 10.0, length = 8.0, mass = 60.0e3, height = 2.8),  # wagons
    (front = 19.0, length = 8.0, mass = 60.0e3, height = 2.8),
]

function train_load(x, xhead)
    q = 0.0
    for car in train
        xf = xhead - car.front # front of this car
        (xf - car.length <= x <= xf) && (q += car.mass * g / car.length)
    end
    return q
end;

# The train load enters the weak form exactly like the dead load, as
# $\int \boldsymbol{\delta u} \cdot \boldsymbol{q} \, \mathrm{d}s$ assembled over the
# roadway bars. Since $q$ jumps at the ends of the cars — usually somewhere in the
# middle of a bar — a Gauss rule made for smooth integrands is a poor fit, so we
# integrate with a composite midpoint rule. This also handles cars that are partly or
# entirely off the bridge for free:
nsub = 20
qr_load = QuadratureRule{RefLine}(fill(2 / nsub, nsub), [Vec(-1 + (2i - 1) / nsub) for i in 1:nsub])
cv_load = CellValues(qr_load, ip^2, ip^2)

function train_loads!(f, dh, cv, xhead)
    fe = zeros(getnbasefunctions(cv))
    for cc in CellIterator(dh, groups["roadway"])
        reinit!(cv, cc)
        fill!(fe, 0.0)
        for qp in 1:getnquadpoints(cv)
            x = spatial_coordinate(cv, qp, getcoordinates(cc))
            q = Vec(0.0, -train_load(x[1], xhead))
            for i in 1:getnbasefunctions(cv)
                fe[i] += (shape_value(cv, qp, i) ⋅ q) * getdetJdV(cv, qp)
            end
        end
        assemble!(f, celldofs(cc), fe)
    end
    return
end;

# ## Drawing the train
#
# For the animation the train itself is written as a handful of quadrilaterals in a
# separate file series. Since the train moves, its *geometry* changes every step —
# something the static-grid VTKHDF format below can not express — so it uses plain
# `.vtu` files collected in a `.pvd`. Each corner also gets a displacement — the
# nodal displacements from [`evaluate_at_grid_nodes`](@ref), interpolated along the
# roadway underneath the corner — so that a warped visualization moves bridge and
# train together:
function write_train(pvd, step, xhead, u)
    un = evaluate_at_grid_nodes(dh, u, :u)
    function deck_displacement(x)
        (0 < x < L) || return zero(Vec{2}) # off the bridge, on solid ground
        i = min(floor(Int, x / a) + 1, length(deck) - 1)
        ξ = x / a - (i - 1)
        return (1 - ξ) * un[deck[i]] + ξ * un[deck[i + 1]]
    end
    points = zeros(3, 4 * length(train))
    disp = zeros(3, 4 * length(train))
    quads = WriteVTK.MeshCell[]
    for (c, car) in enumerate(train)
        xf = xhead - car.front - 0.2 # small gaps between the cars
        xb = xf - car.length + 0.4
        y = 0.4                      # wheel clearance above the roadway
        for (k, (px, py)) in enumerate(((xb, y), (xf, y), (xf, y + car.height), (xb, y + car.height)))
            points[1:2, 4 * (c - 1) + k] .= (px, py)
            disp[1:2, 4 * (c - 1) + k] .= deck_displacement(px)
        end
        push!(quads, WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_QUAD, (4c - 3):(4c)))
    end
    vtk_grid("suspension_bridge_train_$step", points, quads) do vtk
        vtk["u"] = disp
        pvd[xhead] = vtk
    end
    return
end;

# ## Sending the train across
#
# Finally the train rolls over the bridge, from before the first abutment until the
# last wagon has left. Each position is a static load case: dead load plus the axle
# loads, a triangular solve with the already factorized matrix, and stress recovery.
# The bridge fields (displacement and stress) for all steps go into a single temporal
# VTKHDF file:
σ = zeros(getncells(grid))
u = zeros(ndofs(dh))
f = zeros(ndofs(dh))
vtkhdf = VTKHDFGridFile("suspension_bridge.vtkhdf", dh; temporal = true)
pvd = paraview_collection("suspension_bridge_train")
for (step, xhead) in enumerate(-2.0:1.0:90.0) # position of the head of the train
    f .= f_dead
    train_loads!(f, dh, cv_load, xhead)
    apply_rhs!(rhsdata, f, ch)
    u .= Kfact \ f
    bar_stresses!(σ, dh, cv, u)
    write_timestep(vtkhdf, xhead) do vtk
        write_solution(vtk, dh, u)
        write_cell_data(vtk, σ, "stress")
    end
    write_train(pvd, step, xhead, u)
end
close(vtkhdf)
vtk_save(pvd);

# Opening both files in ParaView, warping by the displacement field, and coloring the
# bars by the `stress` cell data (a diverging color map centered at zero neatly
# separates tension from compression) produces the animation at the top of the page:
# the hangers and cables stay in tension, the towers and the deck around them in
# compression, and the girder chords swing back and forth as the train passes.

#md # ## [Plain program](@id suspension_bridge-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here:
#md # [`suspension_bridge.jl`](suspension_bridge.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
