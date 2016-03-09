@testset "Utility testing" begin

@testset "coordxtr + topologyxtr" begin

    Dof = [1 2;
           3 4;
           5 6;
           7 8]'

    Coord = [1.0 3.0;
             2.0 4.0;
             5.0 6.0
             7.0 8.0]'

    Edof = [1 2 3 4 5 6;
            1 2 3 4 7 8]'

    Ex, Ey, Ez = coordxtr(Edof,Coord,Dof, 3)

    @test Ex == [1.0 2.0 5.0;
                  1.0 2.0 7.0]'

    @test Ey == [3.0 4.0 6.0;
                  3.0 4.0 8.0]'

    topo = topologyxtr(Edof,Coord,Dof, 3)

    @test topo == [1 2 3;
                    1 2 4]'
end

end
