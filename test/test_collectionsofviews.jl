@testset "ArrayOfVectorViews" begin
    # Create a vector sorting integers into bins and check
    test_ints = rand(0:99, 100)
    # Create for 3 different sizehints
    aovs = map([20, 1, 100]) do sh
        Ferrite.ArrayOfVectorViews(Int[], (10,); sizehint=sh) do buf
            for v in test_ints
                idx = 1 + v ÷ 10
                Ferrite.push_at_index!(buf, v, idx)
            end
        end
    end
    # Check correct values for the first one
    for (idx, v) in enumerate(aovs[1])
        interval = (10 * (idx-1)):(10 * idx - 1)
        @test all(x -> x ∈ interval, v)
        @test count(x -> x ∈ interval, test_ints) == length(v)
    end
    for aov in aovs
        @test sum(length, aov; init=0) == length(test_ints)
    end
    # Check that the result is independent of sizehint
    for idx in eachindex(aovs[1])
        for aov in aovs[2:end]
            @test aovs[1][idx] == aov[idx]
        end
    end

    # Create an array with random tuple containing and place the tuples
    # according to the values. Check for 2d and 3d arrays.
    for N in 2:3
        tvals = [ntuple(i->rand(0:9), N) for _ in 1:1000]
        aov = Ferrite.ArrayOfVectorViews(NTuple{N, Int}[], (5,5,5)[1:N]; sizehint=10) do buf
            for v in tvals
                idx = 1 .+ v .÷ 2
                Ferrite.push_at_index!(buf, v, idx...)
            end
        end
        @test sum(length, aov; init=0) == length(tvals)
        for (idx, v) in pairs(aov)
            intervals = map(i -> (2 * (i-1)):(2 * i - 1), idx.I)
            @test all(x -> all(map((z, r) -> z ∈ r, x, intervals)), v)
            @test count(x -> all(map((z, r) -> z ∈ r, x, intervals)), tvals) == length(v)
        end
    end
end
