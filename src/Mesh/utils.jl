
"""
function compute_nodal_values(mesh::AbstractMesh, f::Function)
function compute_nodal_values(mesh::AbstractMesh, v::Vector{Int}, f::Function)    
function compute_nodal_values(mesh::AbstractMesh, set::String, f::Function)

Given a `mesh` and some function `f`, `compute_nodal_values` computes all nodal values,
i.e. values at the nodes,  of the function `f`. 
The function implements two dispatches, where only a subset of the mesh's node is used.

```julia
compute_nodal_values(mesh, x -> sin(x[1]) + cos([2]))
compute_nodal_values(mesh, [9, 6, 3], x -> sin(x[1]) + cos([2])) #compute function values at nodes with id 9,6,3
compute_nodal_values(mesh, "right", x -> sin(x[1]) + cos([2])) #compute function values at nodes belonging to nodeset right
```

"""
@inline function compute_nodal_values(nodes::Vector{Node{dim,T}}, f::Function) where{dim,T}
map(n -> f(getcoordinates(n)), nodes)
end

@inline function compute_nodal_values(mesh::AbstractMesh, f::Function)
compute_nodal_values(getnodes(mesh), f::Function)
end

@inline function compute_nodal_values(mesh::AbstractMesh, v::Vector{Int}, f::Function)
compute_nodal_values(getnodes(mesh, v), f::Function)
end

@inline function compute_nodal_values(mesh::AbstractMesh, set::String, f::Function)
compute_nodal_values(getnodes(mesh, set), f::Function)
end

# Transformations
"""
transform!(mesh::Abstractmesh, f::Function)

Transform all nodes of the `mesh` based on some transformation function `f`.
"""
function transform!(g::AbstractMesh, f::Function)
c = similar(g.nodes)
for i in 1:length(c)
    c[i] = Node(f(g.nodes[i].x))
end
copyto!(g.nodes, c)
g
end