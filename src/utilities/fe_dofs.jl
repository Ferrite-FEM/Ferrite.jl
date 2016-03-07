type FEField
    name::AbstractString # Descriptive string
    dim::Int # Number of dimensions for the field
end

type FEDofs
    edof::Matrix{Int} # Element degrees of freedom (This is unnecessary information, can be taken from dof and enodes)
    dof::Matrix{Int} # Degrees of freedom in the nodes
    fieldindex::Vector{Vector{Int}} # Index for the dofs of the fields
    fields::Vector{FEField}
end

# Create a dof_obj given a mesh and set of fields
function FEDofs(fe_m::FEMesh,fe_fields::Vector{FEField})
    # Constructor for FEDofs
    return FEDofs(edof,dof,fieldindex)
end


#############
# Accessors #
#############

# TODO: inline those?

"""
    get_element_dofs(fe_d::FEDofs, elindex::Int) -> el_dofs::Vector
Gets the dofs for a given element index
"""
get_element_dofs(fe_d::FEDofs, elindex::Int) = fe_d.edof[:,elindex]

"""
    get_node_dofs(fe_d::FEDofs, nodeindex::Int) -> node_dofs::Vector
Gets the dofs for a given node index
"""
get_node_dofs(fe_d::FEDofs, nodeindex::Int) = fe_d.dof[:,nodeindex]

"""
    get_field(fe_d::FEDofs,field::Int,u::Vector) -> nodevalues::Matrix
Extracts the specified field from the solution vector u
"""
get_field(fe_d::FEDofs,field::Int,u::Vector) = u[fe_d.edof[fe_d.fieldindex[field],:]]



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