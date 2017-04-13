include("src/JuAFEM.jl")

using Tensors

const dim = 2

function_order = 1
geometry_interpolation = JuAFEM.Lagrange{dim, JuAFEM.RefCube, 1}
function_interpolation = JuAFEM.Lagrange{dim, JuAFEM.RefCube, function_order}()
qr = JuAFEM.QuadratureRule{dim, JuAFEM.RefCube}(function_order + 1)

grid = JuAFEM.generate_grid(geometry_interpolation, (2, 2))

cellvalues = JuAFEM.CellScalarValues(qr, function_interpolation, geometry_interpolation());

dh = JuAFEM.DofHandler(grid)
push!(dh, :T, function_interpolation, 1) # Add a temperature field
JuAFEM.close!(dh)

JuAFEM.get_dof_local_coordinates(function_interpolation)
