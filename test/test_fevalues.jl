import JuAFEM: Square, Triangle


@testset "fevalues" begin

@testset "elasticity_example" begin

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

    function_space = Lagrange{2, JuAFEM.Square, 1}()
    q_rule = get_gaussrule(Dim{2}, Square(), 2)

    fev = FEValues(Float64, q_rule, function_space)


    for cell in 1:size(Edof, 2)
        fill!(Ke, 0)
        ex = Ex[:, cell]
        ey = Ey[:, cell]
        x = reinterpret(Vec{2, Float64}, [ex ey]', (4,))
        reinit!(fev, x)

        for q_point in 1:length(JuAFEM.points(q_rule))

            for i in 1:4
                dNdx = shape_gradient(fev, q_point, i)
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


@testset "function interpolations" begin


    for (function_space, quad_rule) in  ((Lagrange{1, Square, 1}(), get_gaussrule(Dim{1}, Square(), 2)),
                                         (Lagrange{1, Square, 2}(), get_gaussrule(Dim{1}, Square(), 2)),
                                         (Lagrange{2, Square, 1}(), get_gaussrule(Dim{2}, Square(), 2)),
                                         (Lagrange{2, Triangle, 1}(), get_gaussrule(Dim{2}, Triangle(), 2)),
                                         (Lagrange{2, Triangle, 2}(), get_gaussrule(Dim{2}, Triangle(), 2)),
                                         (Lagrange{3, Square, 1}(), get_gaussrule(Dim{3}, Square(), 2)),
                                         (Serendipity{2, Square, 2}(), get_gaussrule(Dim{2}, Square(), 2)),
                                         (Lagrange{3, Triangle, 1}(), get_gaussrule(Dim{3}, Triangle(), 2)))

        fev = FEValues(quad_rule, function_space)
        ndim = n_dim(function_space)
        n_basefuncs = n_basefunctions(function_space)
        x = Vec{ndim, Float64}[rand(Tensor{1,ndim}) for i in 1:n_basefuncs]
        reinit!(fev, x)

        # We test this by applying a given deformation gradient on all the nodes.
        # Since this is a linear deformation we should get back the exact values
        # from the interpolation.
        u = Vec{ndim, Float64}[zero(Tensor{1,ndim}) for i in 1:n_basefuncs]
        u_scal = zeros(n_basefuncs)
        H = rand(Tensor{2, ndim})
        V = rand(Tensor{1, ndim})
        for i in 1:n_basefuncs
            u[i] = H ⋅ x[i]
            u_scal[i] = V ⋅ x[i]
        end

        for i in 1:length(JuAFEM.points(quad_rule))
            @test function_vector_gradient(fev, i, u) ≈ H
            @test function_vector_symmetric_gradient(fev, i, u) ≈ 0.5(H + H')
            @test function_vector_divergence(fev, i, u) ≈ trace(H)
            @test function_scalar_gradient(fev, i, u_scal) ≈ V
            function_scalar_value(fev, i, u_scal)
            function_vector_value(fev, i, u)
        end
    end

end
end
