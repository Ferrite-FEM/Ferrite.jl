# PR #70
function vtk_grid{T}(topology::Matrix{Int}, coord::Matrix{T}, filename::AbstractString)
    Base.depwarn("vtk_grid(topology::Matrix{Int}, coord::Matrix, filename::AbstractString) is deprecated, use vtk_grid{dim,T}(filename::AbstractString, coords::Vector{Vec{dim,T}}, topology::Matrix{Int}, celltype::VTKCellTypes.VTKCellType) instead", :vtk_grid)
    nen = size(topology,1)
    nnodes = size(coord, 2)
    ndim = size(coord, 1)

    if nen == 3 && ndim == 2
        cell =  VTKCellTypes.VTK_TRIANGLE
    elseif nen == 4 && ndim == 2
        cell = VTKCellTypes.VTK_QUAD
    elseif nen == 8 && ndim == 3
        cell = VTKCellTypes.VTK_HEXAHEDRON
    elseif nen == 4 && ndim == 3
        cell = VTKCellTypes.VTK_TETRA
    end

    coords = reinterpret(Vec{ndim,T},coord,(nnodes,))

    vtk = vtk_grid(filename, coords, topology, cell)
    return vtk
end

# Issue #66, PR #73
@deprecate FEValues CellScalarValues

# Issue #74, PR #76
immutable Dim{T} end
export Dim

@deprecate QuadratureRule{dim}(::Type{Dim{dim}}, shape::AbstractRefShape, order::Int) QuadratureRule{dim, typeof(shape)}(order)
@deprecate QuadratureRule{dim}(quad_type::Symbol, ::Type{Dim{dim}}, shape::AbstractRefShape, order::Int) QuadratureRule{dim, typeof(shape)}(quad_type, order)

# Issue #78, PR #80
@deprecate function_scalar_value function_value
@deprecate function_vector_value function_value
@deprecate function_scalar_gradient function_gradient
@deprecate function_vector_gradient function_gradient
@deprecate function_vector_symmetric_gradient function_symmetric_gradient
@deprecate function_vector_divergence function_divergence

# PR #83
@deprecate assemble(edof::Vector, a::Assembler, Ke::Matrix) assemble!(a, Ke, edof)

# PR #85
@deprecate get_quadrule getquadrule
@deprecate get_functionspace getfunctioninterpolation
@deprecate get_geometricspace getgeometryinterpolation
@deprecate detJdV getdetJdV
@deprecate functionspace_n_dim getdim
@deprecate functionspace_ref_shape getrefshape
@deprecate functionspace_order getorder
@deprecate n_boundaries getnfaces
@deprecate functionspace_lower_dim getlowerdim
@deprecate functionspace_lower_order getlowerorder
@deprecate n_basefunctions getnbasefunctions
@deprecate n_boundarynodes getnfacenodes
@deprecate points getpoints
@deprecate weights getweights

# PR #88
@deprecate FECellValues CellScalarValues

# PR #107
@deprecate FunctionSpace Interpolation
@deprecate getfunctionspace getfunctioninterpolation
@deprecate getgeometricspace getgeometryinterpolation
