"""
    BoundaryDofValues([::Type{T},] func_interpol::Interpolation, geom_interpol::Interpolation, boundary_type::Union{Type{<:BoundaryIndex}})

`BoundaryDofValues` stores the information required to apply constraints to the dofs at each boundary entity
(i.e. all facets, faces, edges, or vertices depending on `boundary_type`). What information that is required depends on
the type of function interpolation.

Formally, we need to store information required to evaluate the functionals, lᵢ in ℒ,
defining a "degree of freedom" according to the Ciarlet finite element definition.

For standard scalar or vectorized interpolations, the functionals, lᵢ, are defined as the
value at a fixed coordinate, and only the geometric shape function values at those coordinates
in the reference shape are required.

For vector interpolations, the functionals are typically defined as an integral quantity over the
boundary entity, and further information (such as the size of the entity) is required.
"""
mutable struct BoundaryDofValues{V_GM, FQR}
    const geo_mapping::V_GM # AbstractVector{GeometryMapping}
    const boundary_qr::V_QR # AbstractVector{QuadratureRule}
    current_entity::Int
    const boundary_type::Symbol      # For information only
    const entity_dofs::Vector{Vector{Int}} # Or smth like this - could make getting dofs for specific entities much easier/reusable...
end

function BoundaryDofValues(func_interpol::Interpolation, geom_interpol::Interpolation, boundary_type::Type{<:BoundaryIndex} = FacetIndex)
    return BoundaryDofValues(Float64, func_interpol, geom_interpol, boundary_type)
end

function BoundaryDofValues(::Type{T}, func_interpol::Interpolation{refshape}, geom_interpol::Interpolation{refshape}, boundary_type::Type{<:BoundaryIndex} = FacetIndex) where {T, dim, refshape <: AbstractRefShape{dim}}
    # set up quadrature rules for each boundary entity with dof-positions
    # (determined by func_interpol) as the quadrature points
    interpolation_coords = reference_coordinates(func_interpol)

    # qrs = QuadratureRule{refshape,T,dim}[]
    qrs = map(collect(dirichlet_boundarydof_indices(boundary_type)(func_interpol))) do boundarydofs
        # for boundarydofs in dirichlet_boundarydof_indices(boundary_type)(func_interpol)
        dofcoords = Vec{dim, T}[]
        for boundarydof in boundarydofs
            push!(dofcoords, interpolation_coords[boundarydof])
        end
        QuadratureRule{refshape}(fill(T(NaN), length(dofcoords)), dofcoords) # weights will not be used
        # qrf = QuadratureRule{refshape}(fill(T(NaN), length(dofcoords)), dofcoords) # weights will not be used
        # push!(qrs, qrf)
    end
    geo_mapping = [GeometryMapping{0}(T, geom_interpol, qr) for qr in qrs]

    return BCValues(geo_mapping, fqr, 1)
end

@inline nfacets(bcv::BoundaryDofValues) = length(bcv.geo_mapping)
@inline getcurrentfacet(bcv) = bcv.current_facet

function set_current_facet!(bcv::BoundaryDofValues, facet_nr::Int)
    # Checking facet_nr before setting current_facet allows @inbounds in getcurrentfacet(fv)
    checkbounds(Bool, 1:nfacets(bcv), facet_nr) || throw(ArgumentError("Facet index out of range."))
    return bcv.current_facet = facet_nr
end

@inline get_geo_mapping(bcv::BoundaryDofValues) = @inbounds bcv.geo_mapping[getcurrentfacet(bcv)]
@inline geometric_value(bcv::BoundaryDofValues, q_point::Int, base_func::Int) = geometric_value(get_geo_mapping(bcv), q_point, base_func)
@inline getngeobasefunctions(bcv::BoundaryDofValues) = getngeobasefunctions(get_geo_mapping(bcv))
@inline getnquadpoints(bcv::BoundaryDofValues) = @inbounds getnquadpoints(get_geo_mapping(bcv))
