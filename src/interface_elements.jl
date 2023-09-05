####################################################################################
# Helper functions
####################################################################################


"""
    get_interface_cell_sdim(::Type{<:AbstractRefShape})

Return the spatial dimension of an interface given a base reference shape.
E.g. given `RefTriangle`, 3 is returned, because a triangle forms an interface in 3D.
"""
get_interface_cell_sdim(::Type{RefLine}) = 2
get_interface_cell_sdim(::Type{RefTriangle}) = 3
get_interface_cell_sdim(::Type{RefQuadrilateral}) = 3


"""
    get_interface_cell_shape(::Type{<:AbstractRefShape})

Return the shape of an interface given a base reference shape.
E.g. given `RefTriangle`, `RefPrism` is returned, because two triangles form an interface based on a prism.
"""
get_interface_cell_shape(::Type{RefLine}) = RefQuadrilateral
get_interface_cell_shape(::Type{RefTriangle}) = RefPrism
get_interface_cell_shape(::Type{RefQuadrilateral}) = RefHexahedron


####################################################################################
# Cells
####################################################################################


"""
    InterfaceCell(here::AbstractCell, there::AbstractCell) <: AbstractCell

An `InterfaceCell` is a cell based on two cells of lower dimension representing the two faces.
The two base cells need to use the same reference shape.

# Fields
- `here::AbstractCell`: cell representing the face "here"
- `there::AbstractCell`: cell representing the face "there"
"""
struct InterfaceCell{sdim, rdim, shape<:AbstractRefShape{sdim}, rshape<:AbstractRefShape{rdim}, Chere<:AbstractCell{rshape}, Cthere<:AbstractCell{rshape}} <: AbstractCell{shape}
    here::Chere
    there::Cthere
end

function InterfaceCell(here::Chere, there::Cthere) where {rdim, rshape<:AbstractRefShape{rdim}, Chere<:AbstractCell{rshape}, Cthere<:AbstractCell{rshape}}
    sdim, shape = get_interface_cell_sdim(rshape), get_interface_cell_shape(rshape)
    return InterfaceCell{sdim, rdim, shape, rshape, Chere, Cthere}(here, there)
end

vertices(c::InterfaceCell) = (vertices(c.here)..., vertices(c.there)...)
faces(c::InterfaceCell) = (vertices(c.here), vertices(c.there))
edges(c::InterfaceCell) = (faces(c.here)..., faces(c.there)...)

"""
    get_sides_and_base_indices(c::InterfaceCell)
    get_sides_and_base_indices(chere::AbstractCell, cthere::AbstractCell)
    get_sides_and_base_indices(::Type{<:AbstractRefShape}, ::Type{<:AbstractRefShape})

Return a tuple containing tuples of a symbol (:here or :there) and an integer.
The index of the outer tuple represents the node index.
In the inner tuple, the symbol represents the side the node is on 
and the integer represents the nodes index in the base cell.
"""
get_sides_and_base_indices(c::InterfaceCell) = get_sides_and_base_indices(c.here, c.there)
get_sides_and_base_indices(chere::AbstractCell, cthere::AbstractCell) = get_sides_and_base_indices(typeof(chere), typeof(cthere))

get_sides_and_base_indices(::Type{Line}, ::Type{Line}) = ((:here,1), (:here,2), (:there,1), (:there,2))
get_sides_and_base_indices(::Type{QuadraticLine}, ::Type{Line}) = ((:here,1), (:here,2), (:there,1), (:there,2), (:here,3))
get_sides_and_base_indices(::Type{Line}, ::Type{QuadraticLine}) = ((:here,1), (:here,2), (:there,1), (:there,2), (:there,3))
get_sides_and_base_indices(::Type{QuadraticLine}, ::Type{QuadraticLine}) = ((:here,1), (:here,2), (:there,1), (:there,2), (:here,3), (:there,3))

get_sides_and_base_indices(::Type{Triangle}, ::Type{Triangle}) = ((:here,1), (:here,2), (:here,3), (:there,1), (:there,2), (:there,3))
get_sides_and_base_indices(::Type{QuadraticTriangle}, ::Type{Triangle}) = ((:here,1), (:here,2), (:here,3), (:there,1), (:there,2), (:there,3), (:here,4), (:here,5), (:here,6))
get_sides_and_base_indices(::Type{Triangle}, ::Type{QuadraticTriangle}) = ((:here,1), (:here,2), (:here,3), (:there,1), (:there,2), (:there,3), (:there,4), (:there,5), (:there,6))
get_sides_and_base_indices(::Type{QuadraticTriangle}, ::Type{QuadraticTriangle}) = ((:here,1), (:here,2), (:here,3), (:there,1), (:there,2), (:there,3), (:here,4), (:here,5), (:here,6), (:there,4), (:there,5), (:there,6))

get_sides_and_base_indices(::Type{Quadrilateral}, ::Type{Quadrilateral}) = ((:here,1), (:here,2), (:here,3), (:here,4), (:there,1), (:there,2), (:there,3), (:there,4))
get_sides_and_base_indices(::Type{QuadraticQuadrilateral}, ::Type{Quadrilateral}) = ((:here,1), (:here,2), (:here,3), (:here,4), (:there,1), (:there,2), (:there,3), (:there,4), (:here,5), (:here,6), (:here,7), (:here,8), (:here,9))
get_sides_and_base_indices(::Type{Quadrilateral}, ::Type{QuadraticQuadrilateral}) = ((:here,1), (:here,2), (:here,3), (:here,4), (:there,1), (:there,2), (:there,3), (:there,4), (:there,5), (:there,6), (:there,7), (:there,8), (:there,9))
get_sides_and_base_indices(::Type{QuadraticQuadrilateral}, ::Type{QuadraticQuadrilateral}) = ((:here,1), (:here,2), (:here,3), (:here,4), (:there,1), (:there,2), (:there,3), (:there,4), (:here,5), (:here,6), (:here,7), (:here,8), (:there,5), (:there,6), (:there,7), (:there,8), (:here,9), (:there,9))

function get_node_ids(c::InterfaceCell)
    sni = get_sides_and_base_indices(c)
    return ntuple( i -> getproperty(c, sni[i][1]).nodes[sni[i][2]], length(sni))
    #return  collect( getproperty(c, side).nodes[baseindex] for (side, baseindex) in get_sides_and_base_indices(c) )
end

function Base.getproperty(c::InterfaceCell, s::Symbol)
    if s == :nodes
        return get_node_ids(c)
    else
        return getfield(c,s)
    end
end


####################################################################################
# Interpolation
####################################################################################


# The constructors of `InterpolationInfo` require a `VectorizedInterpolation{InterfaceCellInterpolation}`
# and not an `InterfaceCellInterpolation{VectorizedInterpolation,VectorizedInterpolation}`.
# To create a `VectorizedInterpolation{InterfaceCellInterpolation}`, `InterfaceCellInterpolation` needs to be a `ScalarInterpolation`.

"""
    InterfaceCellInterpolation(here::ScalarInterpolation, there::ScalarInterpolation) <: ScalarInterpolation

An `InterfaceCellInterpolation` is an interpolation based on two interpolations on the faces of an `InterfaceCell`.
If only one interpolation is given, it will be used for both faces.

# Fields
- `here::ScalarInterpolation`: interpolation on the face "here"
- `there::ScalarInterpolation`: interpolation on the face "there"
"""
struct InterfaceCellInterpolation{sdim, rdim, shape<:AbstractRefShape{sdim}, rshape<:AbstractRefShape{rdim},
        IPhere<:ScalarInterpolation{rshape}, IPthere<:ScalarInterpolation{rshape}} <: ScalarInterpolation{shape, Nothing}
    here::IPhere
    there::IPthere
end

function InterfaceCellInterpolation(here::IPhere, there::IPthere) where {
        rdim, rshape<:AbstractRefShape{rdim}, IPhere<:ScalarInterpolation{rshape}, IPthere<:ScalarInterpolation{rshape}}
    sdim, shape = get_interface_cell_sdim(rshape), get_interface_cell_shape(rshape)
    return InterfaceCellInterpolation{sdim, rdim, shape, rshape, IPhere, IPthere}(here, there)
end

function InterfaceCellInterpolation(ip::ScalarInterpolation)
    return InterfaceCellInterpolation(ip, ip)
end

getnbasefunctions(ip::InterfaceCellInterpolation) = getnbasefunctions(ip.here) + getnbasefunctions(ip.there)

"""
    get_n_dofs_on_side(get_dofs::Function, ip::InterfaceCellInterpolation, side::Symbol)

Return the number of DOFs on a `side` (`:here` or `:there`) of an `InterfaceCellInterpolation`.
The function `get_dofs` specifies which DOFs are considered, e.g. by passing `vertexdof_indices`.
"""
function get_n_dofs_on_side(get_dofs::Function, ip::InterfaceCellInterpolation, side::Symbol)
    baseip = getproperty(ip, side)
    return length(get_dofs(baseip)) > 0 ? sum(length.(get_dofs(baseip))) : 0
end
function get_n_dofs_on_side(get_dofs::Function, ip::VectorizedInterpolation{<:Any,<:Any,<:Any,<:InterfaceCellInterpolation}, side::Symbol)
    baseip = getproperty(ip.ip, side)
    return length(get_dofs(baseip)) > 0 ? sum(length.(get_dofs(baseip)))*n_components(ip) : 0
end

"""
    get_interface_index(ip::InterfaceCellInterpolation, side::Symbol, i::Integer)

Return the base function index for an `InterfaceCellInterpolation` given a `side` (`:here` or `:there`)
and the local base function index `i` on that face.
"""
function get_interface_index(ip::InterfaceCellInterpolation, side::Symbol, i::Integer)
    nvhere, nvthere = get_n_dofs_on_side(vertexdof_indices, ip, :here), get_n_dofs_on_side(vertexdof_indices, ip, :there)
    nfhere, nfthere = get_n_dofs_on_side(facedof_interior_indices, ip, :here), get_n_dofs_on_side(facedof_interior_indices, ip, :there)
    nchere, ncthere = get_n_dofs_on_side(celldof_interior_indices, ip, :here), get_n_dofs_on_side(celldof_interior_indices, ip, :there)
    if side == :here
        if i ≤ nvhere
            return i
        elseif i ≤ nvhere + nfhere
            return i + nvthere
        elseif i ≤ nvhere + nfhere + nchere
            return i + nvthere + nfthere
        end
        throw(ArgumentError("No interface index for base index $(i) on side $(side) for interpolation $(ip)."))
    elseif side == :there
        if i ≤ nvthere
            return i + nvhere
        elseif i ≤ nvthere + nfthere
            return i + nvhere + nfhere
        elseif i ≤ nvthere + nfthere + ncthere
            return i + nvhere + nfhere + nchere
        end
        throw(ArgumentError("No interface index for base index $(i) on side $(side) for interpolation $(ip)."))
    end
    throw(ArgumentError("Interface side must be defined by `:here` oder `there`."))
end

"""
    get_interface_dof_indices(get_dofs::Function, ip::InterfaceCellInterpolation) 

Return a tuple of tuples with DOF indices for different entities (vertices, faces, etc.).
The function `get_dofs` specifies which DOFs are considered, e.g. by passing `vertexdof_indices`.
"""
function get_interface_dof_indices(get_dofs::Function, ip::InterfaceCellInterpolation) 
    here  = get_interface_dof_indices(get_dofs, ip, :here)
    there = get_interface_dof_indices(get_dofs, ip, :there)
    return (here..., there...)
end
function get_interface_dof_indices(get_dofs::Function, ip::InterfaceCellInterpolation, side::Symbol)
    basedofs = get_dofs(getproperty(ip, side))
    if isempty(basedofs)
        return (Tuple{}(),)
    else
        return broadcast.(get_interface_index, ((ip,),), ((side,),), basedofs)
    end
end

vertexdof_indices(ip::InterfaceCellInterpolation) = get_interface_dof_indices(vertexdof_indices, ip)

function facedof_indices(ip::InterfaceCellInterpolation)
    here = (get_interface_dof_indices(vertexdof_indices, ip, :here)...,
            get_interface_dof_indices(facedof_interior_indices, ip, :here)...,
            get_interface_dof_indices(celldof_interior_indices, ip, :here)...)
    there = (get_interface_dof_indices(vertexdof_indices, ip, :there)...,
             get_interface_dof_indices(facedof_interior_indices, ip, :there)...,
             get_interface_dof_indices(celldof_interior_indices, ip, :there)...)
    return Tuple(vcat(collect( [t...] for t in here )...)), Tuple(vcat(collect( [t...] for t in there )...))
end

facedof_interior_indices(ip::InterfaceCellInterpolation) = get_interface_dof_indices(celldof_interior_indices, ip)

edgedof_indices(ip::InterfaceCellInterpolation) = get_interface_dof_indices(facedof_indices, ip)

edgedof_interior_indices(ip::InterfaceCellInterpolation) = get_interface_dof_indices(facedof_interior_indices, ip)

function adjust_dofs_during_distribution(ip::InterfaceCellInterpolation)
    return adjust_dofs_during_distribution(ip.here) || adjust_dofs_during_distribution(ip.there) # TODO: Is this really the way to do it?
end

n_components(ip::InterfaceCellInterpolation) = n_components(ip.here)

function default_interpolation(::Type{InterfaceCell{sdim, rdim, shape, rshape, Chere, Cthere}}
    ) where {sdim, rdim, shape<:AbstractRefShape{sdim}, rshape<:AbstractRefShape{rdim},
             Chere<:AbstractCell{rshape}, Cthere<:AbstractCell{rshape}} 
    return InterfaceCellInterpolation(default_interpolation(Chere), default_interpolation(Cthere))
end

function default_geometric_interpolation(ip::InterfaceCellInterpolation{sdim, <:Any, <:Any, <:Any, IPhere, IPthere}
    ) where {sdim, IPhere<:ScalarInterpolation, IPthere<:ScalarInterpolation}
    return InterfaceCellInterpolation(ip.here, ip.there)^sdim
end
function default_geometric_interpolation(ip::InterfaceCellInterpolation{sdim, <:Any, <:Any, <:Any, IPhere, IPthere}
    ) where {sdim, IPhere<:VectorizedInterpolation, IPthere<:VectorizedInterpolation}
    return InterfaceCellInterpolation(ip.here.ip, ip.there.ip)^sdim
end
function default_geometric_interpolation(ip::VectorizedInterpolation{<:Any, <:Any, <:Any, <:InterfaceCellInterpolation})
    return ip
end

getorder(ip::InterfaceCellInterpolation) = getorder(ip.here) == getorder(ip.there) ? getorder(ip.here) : (getorder(ip.here), getorder(ip.there))
getorder(ip::VectorizedInterpolation{<:Any,<:Any,<:Any,<:InterfaceCellInterpolation}) = getorder(ip.ip)

Base.:(^)(ip::InterfaceCellInterpolation, vdim::Int) = VectorizedInterpolation{vdim}(ip)


####################################################################################
# Cell values
####################################################################################


"""
    InterfaceCellValues([::Type{T},] qr::QuadratureRule, func_ip::InterfaceCellInterpolation, [geom_ip::InterfaceCellInterpolation])

An `InterfaceCellValues` wraps two `CellValues`, one for each face of an `InterfaceCell`.

# Fields
- `ip::InterfaceCellInterpolation`: interpolation on the interface
- `here::CellValues`:  values on face "here"
- `there::CellValues`: values on face "there"
- `basefunctionshere::Vector{Int}`: indices for face "here" (used to extract coordinates for base `CellValues`)
- `basefunctionsthere::Vector{Int}`: indices for face "there"
"""
struct InterfaceCellValues{IP, IPhere, IPthere, N_t, dNdx_t, dNdξ_t, T, dMdξ_t, QR, GIPhere, GIPthere} <: AbstractCellValues
    ip::IP
    here::CellValues{IPhere, N_t, dNdx_t, dNdξ_t, T, dMdξ_t, QR, GIPhere}
    there::CellValues{IPthere, N_t, dNdx_t, dNdξ_t, T, dMdξ_t, QR, GIPthere}
    basefunctionshere::Vector{Int}
    basefunctionsthere::Vector{Int}

    function InterfaceCellValues(ip::IP, here::CVhere, there::CVthere) where {IP, IPhere, IPthere, N_t, dNdx_t, dNdξ_t, T, dMdξ_t, QR, GIPhere, GIPthere, 
            CVhere<:CellValues{IPhere, N_t, dNdx_t, dNdξ_t, T, dMdξ_t, QR, GIPhere},
            CVthere<:CellValues{IPthere, N_t, dNdx_t, dNdξ_t, T, dMdξ_t, QR, GIPthere}}
        ip isa VectorizedInterpolation ? sip = ip.ip : sip = ip
        basefunctionshere = collect( get_interface_index(sip, :here, i) for i in 1:getnbasefunctions(sip.here))
        basefunctionsthere = collect( get_interface_index(sip, :there, i) for i in 1:getnbasefunctions(sip.there))
        return new{IP, IPhere, IPthere, N_t, dNdx_t, dNdξ_t, T, dMdξ_t, QR, GIPhere, GIPthere}(ip, here, there, basefunctionshere, basefunctionsthere)
    end
end

function InterfaceCellValues(qr::QuadratureRule,
                             ip::Union{InterfaceCellInterpolation, VectorizedInterpolation{<:Any,<:Any,<:Any,<:InterfaceCellInterpolation}},
                             gip::Union{InterfaceCellInterpolation, VectorizedInterpolation{<:Any,<:Any,<:Any,<:InterfaceCellInterpolation}} = default_geometric_interpolation(ip))
    return InterfaceCellValues(Float64, qr, ip, gip)
end

function InterfaceCellValues(::Type{T}, qr::QuadratureRule, ip::InterfaceCellInterpolation) where {T}
    return InterfaceCellValues(T, qr, ip, default_geometric_interpolation(ip))
end
function InterfaceCellValues(::Type{T}, qr::QuadratureRule, ip::VectorizedInterpolation{<:Any,<:Any,<:Any,<:InterfaceCellInterpolation}) where {T}
    return InterfaceCellValues(T, qr, ip, default_geometric_interpolation(ip.ip))
end

function InterfaceCellValues(::Type{T}, qr::QuadratureRule, 
                             ip::Union{InterfaceCellInterpolation, VectorizedInterpolation{<:Any,<:Any,<:Any,<:InterfaceCellInterpolation}}, 
                             sgip::InterfaceCellInterpolation{sdim}) where {T, sdim}
    return InterfaceCellValues(T, qr, ip, sgip^sdim)
end

function InterfaceCellValues(::Type{T}, qr::QR, ip::IP, gip::VGIP) where {
    T, sdim, rdim, shape <: AbstractRefShape{sdim}, rshape <: AbstractRefShape{rdim},
    QR  <: QuadratureRule{rshape},
    IPhere  <: ScalarInterpolation{rshape},
    IPthere  <: ScalarInterpolation{rshape},
    IP <: InterfaceCellInterpolation{sdim, rdim, shape, rshape, IPhere, IPthere},
    GIPhere <:ScalarInterpolation{rshape},
    GIPthere <:ScalarInterpolation{rshape},
    GIP <: InterfaceCellInterpolation{sdim, rdim, shape, rshape, GIPhere, GIPthere},
    VGIP <: VectorizedInterpolation{sdim, shape, <:Any, GIP}
}
    here = CellValues(T, qr, ip.here, gip.ip.here^sdim)
    there = CellValues(T, qr, ip.there, gip.ip.there^sdim)
    return InterfaceCellValues(ip, here, there)
end

function InterfaceCellValues(::Type{T}, qr::QR, ip::VIP, gip::VGIP) where {
    T, sdim, vdim, rdim, shape <: AbstractRefShape{sdim}, rshape <: AbstractRefShape{rdim},
    QR  <: QuadratureRule{rshape},
    IPhere  <: ScalarInterpolation{rshape},
    IPthere  <: ScalarInterpolation{rshape},
    IP <: InterfaceCellInterpolation{sdim, rdim, shape, rshape, IPhere, IPthere},
    VIP <: VectorizedInterpolation{vdim, shape, <:Any, IP},
    GIPhere <:ScalarInterpolation{rshape},
    GIPthere <:ScalarInterpolation{rshape},
    GIP <: InterfaceCellInterpolation{sdim, rdim, shape, rshape, GIPhere, GIPthere},
    VGIP <: VectorizedInterpolation{sdim, shape, <:Any, GIP}
}
    here = CellValues(T, qr, ip.ip.here^vdim, gip.ip.here^sdim)
    there = CellValues(T, qr, ip.ip.there^vdim, gip.ip.there^sdim)
    return InterfaceCellValues(ip, here, there)
end

reinit!(cv::InterfaceCellValues, cc::CellCache) = reinit!(cv, cc.coords)

function reinit!(cv::InterfaceCellValues, x::AbstractVector{Vec{sdim,T}}) where {sdim, T}
    #nsplit = getngeobasefunctions(cv.here)
    reinit!(cv.here, @view x[cv.basefunctionshere])
    reinit!(cv.there, @view x[cv.basefunctionsthere])
    return nothing
end

getnbasefunctions(cv::InterfaceCellValues) = size(cv.here.N, 1) + size(cv.there.N, 1)

getngeobasefunctions(cv::InterfaceCellValues) = size(cv.here.M, 1) + size(cv.there.N, 1)

getnquadpoints(cv::InterfaceCellValues) = getnquadpoints(cv.here)

"""
    get_side_and_baseindex(cv::InterfaceCellValues, i::Integer)

For an `::InterfaceCellValues`: given the base function index `i` return the side (`:here` or `:there`)
the base function belongs to and the corresponding index for the `CellValues` belonging to that side.
"""
function get_side_and_baseindex(cv::InterfaceCellValues, i::Integer)
    ip = cv.ip
    nvhere, nvthere = get_n_dofs_on_side(vertexdof_indices, ip, :here), get_n_dofs_on_side(vertexdof_indices, ip, :there)
    nfhere, nfthere = get_n_dofs_on_side(facedof_interior_indices, ip, :here), get_n_dofs_on_side(facedof_interior_indices, ip, :there)
    nchere, ncthere = get_n_dofs_on_side(celldof_interior_indices, ip, :here), get_n_dofs_on_side(celldof_interior_indices, ip, :there)
    if i ≤ nvhere
        return :here, i
    elseif i ≤ nvhere + nvthere
        return :there, i - nvhere
    elseif i ≤ nvhere + nvthere + nfhere
        return :here, i - nvthere
    elseif i ≤ nvhere + nvthere + nfhere + nfthere
        return :there, i - nvhere - nfhere
    elseif i ≤ nvhere + nvthere + nfhere + nfthere + nchere
        return :here, i - nvthere - nfthere
    elseif i ≤ nvhere + nvthere + nfhere + nfthere + nchere + ncthere
        return :there, i - nvhere - nfhere - nchere
    end
    throw(ArgumentError("No baseindex for interface index $(i) for interpolation $(ip)."))
end

"""
    get_base_value(get_value::Function, cv::InterfaceCellValues, qp::Int, i::Int, here::Bool)

Return a value from an `::InterfaceCellValues` by specifing:
- `get_value`: function specifing which kind of value, e.g. by passing `shape_value`
- `qp`: index of the quadrature point
- `i`: index of the base function
- `here`: side of the interface, where `true` means "here" and `false` means "there".
"""
function get_base_value(get_value::Function, cv::InterfaceCellValues, qp::Int, i::Int, here::Bool)
    side, baseindex = get_side_and_baseindex(cv, i)
    if side == :here && here
        return get_value(cv.here, qp, baseindex)
    elseif side == :there && ! here
        return get_value(cv.there, qp, baseindex)
    else
        return 0
    end
end

"""
    shape_value(cv::InterfaceCellValues, qp::Int, i::Int, here::Bool)

Return the value of shape function `i` evaluated in quadrature point `qp`
on side `here`, where `true` means "here" and `false` means "there".
"""
shape_value(cv::InterfaceCellValues, qp::Int, i::Int, here::Bool) = get_base_value(shape_value, cv, qp, i, here)

"""
    shape_gradient(cv::InterfaceCellValues, qp::Int, i::Int, here::Bool)

Return the gradient of shape function `i` evaluated in quadrature point `qp`
on side `here`, where `true` means "here" and `false` means "there".
"""
shape_gradient(cv::InterfaceCellValues, qp::Int, i::Int, here::Bool) = get_base_value(shape_gradient, cv, qp, i, here)

"""
    shape_value_average(cv::InterfaceCellValues, qp::Int, i::Int)

Return the value of shape function `i` evaluated in quadrature point `qp`
for computing the average value on an interface.
"""
function shape_value_average(cv::InterfaceCellValues, qp::Int, i::Int)
    side, baseindex = get_side_and_baseindex(cv, i)
    return side == :here ? shape_value(cv.here, qp, baseindex) / 2 : shape_value(cv.there, qp, baseindex) / 2
end

"""
    shape_gradient_average(cv::InterfaceCellValues, qp::Int, i::Int)

Return the gradient of shape function `i` evaluated in quadrature point `qp`
for computing the average gradient on an interface.
"""
function shape_gradient_average(cv::InterfaceCellValues, qp::Int, i::Int)
    side, baseindex = get_side_and_baseindex(cv, i)
    return side == :here ? shape_gradient(cv.here, qp, baseindex) / 2 : shape_gradient(cv.there, qp, baseindex) / 2
end

"""
    shape_value_jump(cv::InterfaceCellValues, qp::Int, i::Int)

Return the value of shape function `i` evaluated in quadrature point `qp`
for computing the value jump on an interface.
"""
function shape_value_jump(cv::InterfaceCellValues, qp::Int, i::Int)
    side, baseindex = get_side_and_baseindex(cv, i)
    return side == :here ? -shape_value(cv.here, qp, baseindex) : shape_value(cv.there, qp, baseindex)
end

"""
    shape_gradient_jump(cv::InterfaceCellValues, qp::Int, i::Int)

Return the gradient of shape function `i` evaluated in quadrature point `qp`
for computing the gradient jump on an interface.
"""
function shape_gradient_jump(cv::InterfaceCellValues, qp::Int, i::Int)
    side, baseindex = get_side_and_baseindex(cv, i)
    return side == :here ? -shape_gradient(cv.here, qp, baseindex) : shape_gradient(cv.there, qp, baseindex)
end