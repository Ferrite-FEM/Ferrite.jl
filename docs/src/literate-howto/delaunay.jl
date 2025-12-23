# # [Mesh generation with DelaunayTriangulation.jl](@id howto-delaunay)

#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`delaunay.ipynb`](@__NBVIEWER_ROOT_URL__/howto/delaunay.ipynb).
#-
#
# ## Introduction 
#
# In this example, we will demonstrate how to generate a mesh using the 
# [DelaunayTriangulation.jl](https://github.com/JuliaGeometry/DelaunayTriangulation.jl).
# This package allows for the generation of a Delaunay triangulation of a set of points, boundaries,
# and constrained edges using `triangulate`, and mesh refinement is done using `refine!`. We then
# show how to convert this mesh into a Ferrite grid.

# ## Implementation 
# We first load the packages we will need.
using Ferrite, DelaunayTriangulation, OrderedCollections, SparseArrays
import CairoMakie: triplot, tricontourf

# ### Mesh generation
# Next, we need to define a mesh for $\Omega$ using DelaunayTriangulation.jl. The annulus can be 
# created by making two `DelaunayTriangulation.CircularArc`s. The syntax for `CircularArc` is 
# `CircularArc(p, q, c)`, where `p` and `q` are two points on the circle, and `c` is the center of the circle.
function Annulus(R₁, R₂) # center at (0, 0) 
    inner = CircularArc((R₁, 0.0), (R₁, 0.0), (0.0, 0.0), positive=false)
    outer = CircularArc((R₂, 0.0), (R₂, 0.0), (0.0, 0.0))
    return [[[outer]], [[inner]]]
end
#md nothing # hide

# The need for `positive = false` is so that the domain boundary is positively oriented, as explained 
# [here](https://juliageometry.github.io/DelaunayTriangulation.jl/stable/manual/boundaries/). The mesh 
# is then generated using `triangulate`.
points = NTuple{2,Float64}[]
R₁, R₂ = 1.0, 2.0
boundary_nodes = Annulus(R₁, R₂)
tri = triangulate(points; boundary_nodes)

# As explained in DelaunayTriangulation.jl's [curve-bounded refinement tutorial](https://juliageometry.github.io/DelaunayTriangulation.jl/stable/tutorials/curve_bounded/),
# `triangulate` only creates an initial coarse discretisation of the boundary with no interior nodes. We improve the mesh by using `refine!`.
refine!(tri, max_area=1e-3π * (R₂^2 - R₁^2)) 

# Custom constraints beyond simple scalar constraints are also possible, as discussed in [this DelaunayTriangulation.jl tutorial](https://juliageometry.github.io/DelaunayTriangulation.jl/dev/tutorials/refinement/#Constrained-triangulation-and-custom-constraints),
# which could allow for e.g. heterogeneous meshes or adaptive mesh refinement.

# The mesh that we have now can be easily visualised using `triplot` from Makie.
triplot(tri)
# *Figure 1*: Delaunay triangulation of the annulus $\Omega = \{\textbf{x} \in \mathbb{R}^2 : 1 < \|\textbf{x}\| < 2\}$.

# ### Converting to a Ferrite grid
# Now we need to convert `tri` into a `Grid`. For this, we need to get cells, nodes, and 
# boundary information. The following function collects the cells; for information about the 
# `solid` adjective, see the DelaunayTriangulation.jl documentation [here](https://juliageometry.github.io/DelaunayTriangulation.jl/stable/manual/ghost_triangles/)
# about ghost vertices.
function get_cells(tri::Triangulation)
    cells = Vector{Triangle}(undef, num_solid_triangles(tri))
    cell_map = Dict{NTuple{3,Int},Int}()
    for (cell_number, triangle) in enumerate(each_solid_triangle(tri))
        ijk = triangle_vertices(triangle)
        uvw = DelaunayTriangulation.sort_triangle(ijk)
        cells[cell_number] = Triangle(uvw)
        cell_map[uvw] = cell_number
    end
    return cells, cell_map
end
#md nothing # hide

# The `sort_triangle` function rotates a triangle `(i, j, k)` so that the last vertex is the smallest, while 
# preserving the orientation. This allows us to not have to check whether the triangle we have is stored 
# as `(i, j, k)`, `(j, k, i)`, or `(k, i, j)`. The `cell_map` is needed to easily find a triangle in the 
# `cells` vector. The nodes can be obtained using `get_nodes` below.
function get_nodes(tri::Triangulation)
    nodes = Vector{Node{2,Float64}}(undef, DelaunayTriangulation.num_points(tri))
    for vertex in DelaunayTriangulation.each_point_index(tri)
        p = get_point(tri, vertex)
        x, y = getxy(p)
        nodes[vertex] = Node((x, y))
    end
    return nodes
end
#md nothing # hide

# We iterate over `each_point_index` here rather than `each_solid_vertex` since not all points may 
# be in the triangulation, but we still need to store them to make sure the vertices in `cells`
# match the `nodes` vector. To now obtain the boundary information, some care is needed. 
# We will obtain the boundary information such that the boundary edges are listed consecutively,
# which complicates the implementation slightly; unordered iteration over the boundary can be done 
# using `DelaunayTriangulation.each_boundary_edge(tri)` or using `get_boundary_edge_map(tri)`. Our 
# implementation is below.
function to_facetindex(cell_map, ijk, e)
    uvw = DelaunayTriangulation.sort_triangle(ijk)
    cell_number = cell_map[uvw]
    u, v, w = triangle_vertices(uvw)
    i, j = edge_vertices(e)
    (i, j) == (u, v) && return FacetIndex(cell_number, 1)
    (i, j) == (v, w) && return FacetIndex(cell_number, 2)
    return FacetIndex(cell_number, 3) # assume (i, j) == (w, u)
end
function get_facetsets(tri::Triangulation, cell_map)
    facetsets = Dict{String,OrderedSet{FacetIndex}}()
    g = 0
    nc = DelaunayTriangulation.num_curves(tri)
    for curve_index in 1:nc
        if nc == 1
            bn = get_boundary_nodes(tri)
        else
            bn = get_boundary_nodes(tri, curve_index)
        end
        ns = DelaunayTriangulation.num_sections(bn)
        for segment_index in 1:ns
            g -= 1
            set = OrderedSet{FacetIndex}()
            facetsets[string(g)] = set
            if nc == ns == 1
                bnn = bn
            else
                bnn = get_boundary_nodes(bn, segment_index)
            end
            ne = num_boundary_edges(bnn)
            for i in 1:ne
                u = get_boundary_nodes(bnn, i)
                v = get_boundary_nodes(bnn, i + 1)
                w = get_adjacent(tri, u, v)
                push!(set, to_facetindex(cell_map, (u, v, w), (u, v)))
            end
        end
    end
    return facetsets
end
#md nothing # hide

# The `nc == 1` and `nc == ns == 1` checks are not used for the mesh we have generated, but they are needed 
# for simpler meshes. The `g` variable is used to keep track of the associated ghost vertex. We now 
# have all that we need to generate our `Grid`.
function create_ferrite_grid(tri::Triangulation)
    cells, cell_map = get_cells(tri)
    nodes = get_nodes(tri)
    facetsets = get_facetsets(tri, cell_map)
    return Grid(cells, nodes; facetsets)
end
grid = create_ferrite_grid(tri)

# The `facetsets` in this case have the ghost vertices as the keys:
Ferrite.getfacetsets(grid)

# The `-1` ghost vertex refers to the outer boundary, while the `-2` ghost vertex refers to the inner boundary.
# The ordering of these ghost vertices comes from the ordering of sections in the `boundary_nodes` variable.

# ### Solving the problem
# By transforming the Delaunay triangulation into a Ferrtie `Grid`, we are now in a position to solve 
# finite element problems using the standard Ferrite workflow; see e.g. the [heat flow](@ref tutorial-heat-equation) and [linear elasticity](@ref tutorial-linear-elasticity) tutorials.
# The main difference is the need to use triangles for defining the interpolation and quadrature rules. For illustration,
# here we define a scalar field interpolation describing an analytic solution that is simply the $\sqrt{x^2 + y^2}$.
dh = DofHandler(grid)  
add!(dh, :u, Lagrange{RefTriangle,1}())  
close!(dh)  
a = zeros(DelaunayTriangulation.num_points(tri))
apply_analytical!(a, dh, :u, norm)
u_nodes = evaluate_at_grid_nodes(dh, a, :u)

# ### Plotting
# The solution can be visualised as in previous tutorials. Alternatively, we can use `tricontourf` which 
# knows how to use DelaunayTriangulation.jl's `Triangulation`.
tricontourf(tri, u_nodes)
# *Figure 2*: Solution on the domain $\Omega = \{\textbf{x} \in \mathbb{R}^2 : 1 < \|\textbf{x}\| < 2\}$.

#md # ## [Plain program](@id delaunay-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`delaunay.jl`](delaunay.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
