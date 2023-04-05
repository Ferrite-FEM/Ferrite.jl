# See https://defelement.com/ciarlet.html, J= dFdx. Seems like only the following three tensors/scalars are required
# det(J), J, and J⁻¹ (all which are currently calculated). The only question is if passing them around infers extra costs. 


struct GeoQuadValues{dim,T<:Real,RefShape<:AbstractRefShape}
    M::Matrix{T}
    dMdξ::Matrix{Vec{dim,T}}
    #dM²d²ξ # Required for Piola mappings?
    ip::Interpolation{dim,RefShape}
end
function GeoQuadValues(cv::Union{CellVectorValues,CellScalarValues})
    return GeoQuadValues(cv.M, cv.dMdξ, cv.geo_interp)
end

abstract type FunctionQuadValues{dim,T,RefShape} <: Values{dim,T,RefShape} end

struct FunctionQuadScalarValues{dim,T<:Real,RefShape<:AbstractRefShape} <: FunctionQuadValues{dim,T,RefShape}
    # Constant (precalculated values)
    Nξ::Matrix{T} 
    dNdξ::Matrix{Vec{dim,T}}
    # To be updated for each quad point
    #Nx::Vector{T} # Probably not required for scalar shape functions
    dNdx::Vector{Vec{dim,T}} # For current quadrature point    
    # Just for information, don't use in performance critical code 
    ip::Interpolation{dim,RefShape}
end
FieldTrait(::Type{<:FunctionScalarValues}) = ScalarValued()

struct FunctionQuadVectorValues{dim,T<:Real,RefShape<:AbstractRefShape,M} <: FunctionQuadValues{dim,T,RefShape}
    # Constant (precalculated values)
    Nξ::Matrix{Vec{dim,T}} # For greater generality, I think Nξ (constant) and Nx are needed for non-identity mappings (but only vector values)
    dNdξ::Matrix{Tensor{2,dim,T,M}}
    # To be updated for each quad point
    #Nx::Vector{Vec{dim,T}} # For H(curl) and H(div) I think this would be required. 
    dNdx::Vector{Tensor{2,dim,T,M}} # Update for current quadrature point
    ip::Interpolation{dim,RefShape}
end
FieldTrait(::Type{<:FunctionVectorValues}) = VectorValued()

mutable struct CellQuadValues{dim,T,RefShape,FVS<:NamedTuple, Mapping} <: CellValues{dim,T,RefShape}
    const geo_values::GeoQuadValues{dim,T,RefShape,Mapping}
    const fun_values::FVS 
    const qr::QuadratureRule{dim,RefShape,T}
    const x::Vector{Vec{dim,T}}
    detJdV::T
    q_point::Int
end