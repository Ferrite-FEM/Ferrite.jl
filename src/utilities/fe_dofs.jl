type FEField
    name::UTF8String # Descriptive string
    dim::Int # Number of dimensions for the field
end

type FEDofs
    mesh::FEMesh
    dof::Matrix{Int} # Degrees of freedom in the nodes
    fieldindex::Vector{Vector{Int}} # Index for the dofs of the fields
    fields::Vector{FEField}
end

"""
    FEDofs(fe_m::FEMesh,fe_f::Vector{FEField}) -> fe_d::FEDofs
Sets up a dof object with dof numbering for specified fields
"""
function FEDofs(fe_m::FEMesh,fe_f::Vector{FEField})

    n_fields = length(fe_f)
    n_subfields = sum([fe_f[i].dim for i = 1:n_fields])
    n_nodes = maximum(fe_m.topology)

    dof = Int[]

    for i = 1:n_nodes
        append!(dof,(i-1)*n_subfields+1:i*n_subfields)
    end

    dof = reshape(dof,(n_subfields,n_nodes))

    fieldindex = Vector{Int}[]
    for i = 1:n_fields
        push!(fieldindex,collect(1:fe_f[i].dim) + sum([fe_f[j].dim for j = 1:(i-1)]))
    end

    return FEDofs(fe_m,dof,fieldindex,fe_f)
end


#############
# Accessors #
#############

"""
    get_element_dofs(fe_d::FEDofs, elindex::Int) -> el_dofs::Vector
Gets the dofs for a given element index
"""
@inline function get_element_dofs(fe_d::FEDofs, elindex::Int)
    edof = Int[]
    for i = 1:length(fe_d.fields)
        append!(edof,fe_d.dof[fe_d.fieldindex[i],fe_d.mesh.topology[:,elindex]][:])
    end
    return edof
end

"""
    get_node_dofs(fe_d::FEDofs, nodeindex::Int) -> node_dofs::Vector
Gets the dofs for a given node index
"""
@inline get_node_dofs(fe_d::FEDofs, nodeindex::Int) = fe_d.dof[:,nodeindex]

"""
    get_field(fe_d::FEDofs,field::Int,u::Vector) -> nodevalues::Matrix
Extracts the specified field from the solution vector u
"""
@inline get_field(fe_d::FEDofs,field::Int,u::Vector) = u[fe_d.dof[fe_d.fieldindex[field],:]]

# Duplicates from FEMesh (maybe these are the only ones needed, but can be nice to access this information from FEMesh too)
"""
    get_element_nodes(fe_d::FEDofs, elindex::Int) -> el_nodes::Vector
Gets the nodes for a given element index
"""
@inline get_element_nodes(fe_d::FEDofs, elindex::Int) = fe_d.mesh.topology[:,elindex]

"""
    get_element_coords(fe_d::FEDofs, elindex::Int) -> elcoord::Vector{Tensor{1,dim}}
Gets the coordinates of the nodes for a given element index
"""
@inline get_element_coords(fe_d::FEDofs, elindex::Int) = fe_d.mesh.coords[fe_d.mesh.topology[:,elindex]]

"""
    get_node_coords(fe_d::FEDofs, nodeindex::Int) -> nodecoord::Tensor{1,dim}
Gets the coordinates of the nodes for a given element index
"""
@inline get_node_coords(fe_d::FEDofs, nodeindex::Int) = fe_d.mesh.coords[nodeindex]

"""
    get_boundary_nodes(fe_d::FEDofs, boundaryindex::Int) -> boundarynodes::Vector
Gets the nodes on a specified boundary
"""
@inline get_boundary_nodes(fe_d::FEMesh, boundaryindex::Int) = fe_d.mesh.boundary[boundaryindex]



################
# Enhancements #
################

# TODO: Should those be ! functions that update the FEDofs object?

"""
    add_field(fe_d::FEDofs,fe_f::FEField)
Adds a new field to fe_d
"""
function add_field(fe_d::FEDofs,fe_f::FEField)
    # Add another field to the FEDofs object
end

"""
    remove_field(fe_d::FEDofs,fe_f::FEField)
Removes a field from fe_d
"""
function remove_field(fe_d::FEDofs,fe_f::FEField)
    # Remove a fiel from the FEDofs object
end