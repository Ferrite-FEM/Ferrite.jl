abstract ShapeFunc

immutable Lagrange_Square_1 <: ShapeFunc end
immutable Lagrange_Tri_1 <: ShapeFunc end
immutable Serend_Square_2 <: ShapeFunc end
immutable Lagrange_Cube_1 <: ShapeFunc end

abstract GeoShape
abstract GeoShape_3D <: GeoShape
abstract GeoShape_2D <: GeoShape
abstract GeoShape_1D <: GeoShape

get_ndim(::GeoShape_3D) = 3
get_ndim(::GeoShape_2D) = 2
get_ndim(::GeoShape_1D) = 1

immutable Line <: GeoShape_1D end
immutable Triangle <: GeoShape_2D end
immutable Square <: GeoShape_2D end
immutable Cube <: GeoShape_3D end
immutable Tetrahedra <: GeoShape_3D end

# TODO document this
immutable FElement
    name::Symbol
    shape::GeoShape
    shape_funcs::ShapeFunc
    inits::Dict{Symbol, NTuple{2, Int}}
    nnodes::Int
    dofs_per_node::Int
    flux_size::Int
    grad_kernel::Function
    source_kernel::Function
    flux_kernel::Function
    intf_kernel::Function
    default_intorder::Int # Should give a full rank stiffness matrix
end

get_ndim(fele::FElement) = get_ndim(fele.shape)
get_nflux(fele::FElement) = fele.flux_size

function show(io::IO, fele::FElement)
    print(io, "Name: $(fele.name)\n")
    print(io, "Reference shape: $(typeof(fele.shape))\n")
    print(io, "Shape functions: $(typeof(fele.shape_funcs))\n")
    print(io, "Number of nodes: $(fele.nnodes)\n")
    print(io, "Dofs per node: $(fele.dofs_per_node)")
end
