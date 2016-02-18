using JuAFEM
import JuAFEM: Square

@testset "fevalues" begin

    x = [0.0 1.0 1.5 0.5;
         0.0 0.2 0.8 0.6]


    Coord = Float64[0 0
                    1 0
                    1 1
                    0 1
                    2 0
                    2 1
                    2 2
                    1 2
                    0 2]'

    Dof = [1 2
           3 4
           5 6
           7 8
           9 10
           11 12
           13 14
           15 16
           17 18]'

    Edof = [1 2 3 4 5 6 7 8;
            3 4 9 10 11 12 5 6;
            5 6 11 12 13 14 15 16;
            7 8 5 6 15 16 17 18]'

    Ex, Ey, Ez = coordxtr(Edof, Coord, Dof, 4)

    a = start_assemble()
    Ke = zeros(8,8)
    Ke_2 = zeros(8,8)
    B = zeros(4, 8)
    D = hooke(2, 200e9, 0.3)
    t = 2.0

    function_space = JuAFEM.Lagrange{1, JuAFEM.Square}()
    q_rule = JuAFEM.get_gaussrule(JuAFEM.Square(), 2)
    fev = FEValues(Float64, q_rule, function_space)


    for cell in 1:size(Edof, 2)
        fill!(Ke, 0)
        ex = Ex[:, cell]
        ey = Ey[:, cell]
        x = [ex ey]'
        reinit!(fev, x)

        for q_point in 1:length(JuAFEM.points(q_rule))

            for i in 1:4
                dNdx = shape_derivative(fev, q_point, i)
                B[1, 2*i - 1] = dNdx[1]
                B[2, 2*i - 0] = dNdx[2]
                B[4, 2*i - 0] = dNdx[1]
                B[4, 2*i - 1] = dNdx[2]
            end


            Ke += B' * (D * B) * detJdV(fev, q_point) * t
        end

        Ke_2, _ = plani4e(ex, ey, [2, 2, 2], D)

        @test norm(Ke - Ke_2) / norm(Ke) ≈ 0.0
    end
end