export DirichletBoundaryConditions, update!, apply!, add!

"""
    DirichletBoundaryConditions

A dirichlet boundary conditions is a a boundary where a dof is fixed to take a certain value.
The struct `DirichletBoundaryConditions` represents a collection of such boundary conditions.

It is created from a `DofHandler`

```jldoctest dbc
julia> dbc = DirichletBoundaryConditions(dh)
```

Dirichlet boundary conditions are added to certain components of a field for a specific nodes of the grid.
A function is also given that should be of the form `(x,t) -> v` where `x` is the coordinate of the node, `t` is a
time parameter and `v` should be of the same length as the number of components the bc is applied to:

```jldoctest
julia> addnodeset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0);

julia> nodes = grid.nodesets["clamped"]

julia> push!(dbc, :temperature, nodes, (x,t) -> t * [x[2], 0.0, 0.0], [1, 2, 3])
```

Boundary conditions are now updates by specifying a time:

```jldoctest
julia> t = 1.0;

julia> update!(dbc, t)
```

The boundary conditions can be applied to a vector with

```jldoctest
julia> u = zeros(ndofs(dh))

julia> apply!(u, dbc)
```

"""
immutable DirichletBoundaryCondition
    f::Function
    nodes::Set{Int}
    field::Symbol
    components::Vector{Int}
    idxoffset::Int
end


immutable DirichletBoundaryConditions{DH <: DofHandler}
    bcs::Vector{DirichletBoundaryCondition}
    dofs::Vector{Int}
    values::Vector{Float64}
    dh::DH
    closed::Ref{Bool}
end

function DirichletBoundaryConditions(dh::DofHandler)
    @assert isclosed(dh)
    DirichletBoundaryConditions(DirichletBoundaryCondition[], Int[], Float64[], dh, Ref(false))
end

function Base.show(io::IO, dbcs::DirichletBoundaryConditions)
    println(io, "DirichletBoundaryConditions:")
    if !isclosed(dbcs)
        print(io, "  Not closed!")
    else
        println(io, "  BCs:")
        for dbc in dbcs.bcs
            println(io, "    ", "Field: ", dbc.field, " ", "Components: ", dbc.components)
        end
    end
end

isclosed(dbcs::DirichletBoundaryConditions) = dbcs.closed[]
dirichlet_dofs(dbcs::DirichletBoundaryConditions) = dbcs.dofs
free_dofs(dbcs::DirichletBoundaryConditions) = setdiff(dbcs.dh.dofs_nodes, dbcs.dofs)
function close!(dbcs::DirichletBoundaryConditions)
    fill!(dbcs.values, NaN)
    dbcs.closed[] = true
    return dbcs
end

function add!(dbcs::DirichletBoundaryConditions, field::Symbol,
                          nodes::Union{Set{Int}, Vector{Int}}, f::Function, component::Int=1)
    add!(dbcs, field, nodes, f, [component])
end

function add!(dbcs::DirichletBoundaryConditions, field::Symbol,
                          nodes::Union{Set{Int}, Vector{Int}}, f::Function, components::Vector{Int})
    @assert field in dbcs.dh.field_names || error("Missing: $field")
    for component in components
        @assert 0 < component <= ndim(dbcs.dh, field)
    end

    if length(nodes) == 0
        warn("Added Dirichlet BC to node set containing 9 nodes")
    end

    dofs_bc = Int[]
    offset = dof_offset(dbcs.dh, field)
    for node in nodes
        for component in components
            push!(dofs_bc, dbcs.dh.dofs_nodes[offset + component, node])
        end
    end

    n_bcdofs = length(dofs_bc)

    append!(dbcs.dofs, dofs_bc)
    idxoffset = length(dbcs.values)
    resize!(dbcs.values, length(dbcs.values) + n_bcdofs)

    push!(dbcs.bcs, DirichletBoundaryCondition(f, Set(nodes), field, components, idxoffset))

end

function update!(dbcs::DirichletBoundaryConditions, time::Float64 = 0.0)
    @assert dbcs.closed[]
    bc_offset = 0
    for dbc in dbcs.bcs
        # Function barrier
        _update!(dbcs.values, dbc.f, dbc.nodes, dbc.field,
                              dbc.components, dbcs.dh, dbc.idxoffset, time)
    end
end

function _update!(values::Vector{Float64}, f::Function, nodes::Set{Int}, field::Symbol,
                              components::Vector{Int}, dh::DofHandler, idx_offset::Int, time::Float64)
    mesh = dh.grid
    offset = dof_offset(dh, field)
    current_dof = 1
     for node in nodes
        x = getcoordinates(getnodes(mesh, node))
        bc_value = f(x, time)
        @assert length(bc_value) == length(components)
        for i in 1:length(components)
            values[current_dof + idx_offset] = bc_value[i]
            current_dof += 1
        end
    end
end

function vtk_point_data(vtkfile, dbcs::DirichletBoundaryConditions)
    unique_fields = []
    for dbc in dbcs.bcs
        push!(unique_fields, dbc.field)
    end
    unique_fields = unique(unique_fields)

    for field in unique_fields
        nd = ndim(dbcs.dh, field)
        data = zeros(Float64, nd, getnnodes(dbcs.dh.grid))
        for dbc in dbcs.bcs
            if dbc.field != field
                continue
            end

            for node in dbc.nodes
                for component in dbc.components
                    data[component, node] = 1.0
                end
            end
        end
        vtk_point_data(vtkfile, data, string(field)*"_bc")
    end
    return vtkfile
end

function apply!(v::Vector, bc::DirichletBoundaryConditions)
    @assert length(v) == ndofs(bc.dh)
    v[bc.dofs] = bc.values
    return v
end
