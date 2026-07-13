# Test the linear shell tutorial script and its through-thickness integration.
module TestLinearShell
using LinearAlgebra, Test

mktempdir() do dir
    cd(dir) do
        include(joinpath(@__DIR__, "../../docs/src/literate-tutorials/linear_shell.jl"))
    end
end

ip = Lagrange{RefQuadrilateral, 1}()
qr_inplane = QuadratureRule{RefQuadrilateral}(1)
qr_ooplane = QuadratureRule{RefLine}(2)
cv = CellValues(qr_inplane, ip, ip^3)
x = Vec{3, Float64}[
    Vec((-1.0, -1.0, 0.0)), Vec((1.0, -1.0, 0.0)),
    Vec((1.0, 1.0, 0.0)), Vec((-1.0, 1.0, 0.0)),
]
reinit!(cv, x)

ke_thin = zeros(20, 20)
ke_thick = zeros(20, 20)
integrate_shell!(ke_thin, cv, qr_ooplane, x, (thickness = 1.0, C = Matrix{Float64}(I, 5, 5)))
integrate_shell!(ke_thick, cv, qr_ooplane, x, (thickness = 2.0, C = Matrix{Float64}(I, 5, 5)))

# The translational membrane block is independent of the through-thickness
# coordinate, so its integral must scale linearly with thickness.
@test ke_thick[1:12, 1:12] ≈ 2 * ke_thin[1:12, 1:12]
end
