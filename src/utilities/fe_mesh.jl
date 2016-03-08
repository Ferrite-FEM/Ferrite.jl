# Typ definitions
type FEMesh{dim, T <: Real}
    coords::Vector{Vec{dim,T}} # Node coordinates
    topology::Matrix{Int} # Element nodes
    boundary::Vector{Vector{Int}} # Boundary nodes
end

# function FEMesh()
#     # Set up the mesh here
#     return FEMesh(coords,topology,boundary)
# end


#############
# Accessors #
#############

# Should these even exist? Atleast I should add the same accessors for the FEDofs aswell so that contains all information needed

"""
    get_element_nodes(fe_m::FEMesh, elindex::Int) -> el_nodes::Vector
Gets the nodes for a given element index
"""
@inline get_element_nodes(fe_m::FEMesh, elindex::Int) = fe_m.topology[:,elindex]

"""
    get_element_coords(fe_m::FEMesh, elindex::Int) -> elcoord::Vector{Tensor{1,dim}}
Gets the coordinates of the nodes for a given element index
"""
@inline get_element_coords(fe_m::FEMesh, elindex::Int) = fe_m.coords[fe_m.topology[:,elindex]]

"""
    get_node_coords(fe_m::FEMesh, nodeindex::Int) -> nodecoord::Tensor{1,dim}
Gets the coordinates of the nodes for a given element index
"""
@inline get_node_coords(fe_m::FEMesh, nodeindex::Int) = fe_m.coords[nodeindex]

"""
    get_boundary_nodes(fe_m::FEMesh, boundaryindex::Int) -> boundarynodes::Vector
Gets the nodes on a specified boundary
"""
@inline get_boundary_nodes(fe_m::FEMesh, boundaryindex::Int) = fe_m.boundary[boundaryindex]