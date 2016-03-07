# Typ definitions
type FEMesh{dim, T <: Real}
    coords::Vector{Vec{dim,T}} # Node coordinates
    enodes::Matrix{Int} # Element nodes
    boundary::Vector{Vector{Int}} # Boundary nodes
end

function FEMesh()
    # Set up the mesh here
    return FEMesh(coords,enodes,boundary)
end


#############
# Accessors #
#############

# TODO: inline those?

"""
    get_element_nodes(fe_m, elindex::Int) -> el_nodes::Vector
Gets the nodes for a given element index
"""
get_element_nodes(fe_m::FEMesh, elindex::Int) = fe_m.enodes[:,elindex]

"""
    get_element_coords(fe_m, elindex::Int) -> elcoord::Vector{Tensor{1,dim}}
Gets the coordinates of the nodes for a given element index
"""
get_element_coords(fe_m::FEMesh, elindex::Int) = fe_m.coords[fe_m.enodes[:,elindex]]

"""
    get_node_coords(fe_m, nodeindex::Int) -> nodecoord::Tensor{1,dim}
Gets the coordinates of the nodes for a given element index
"""
get_node_coords(fe_m::FEMesh, nodeindex::Int) = fe_m.coords[nodeindex]

"""
    get_boundary_nodes(fe_m, boundaryindex::Int) -> boundarynodes::Vector
Gets the nodes on a specified boundary
"""
get_boundary_nodes(fe_m::FEMesh, boundaryindex::Int) = fe_m.boundary[boundaryindex]