facts("Spring testing") do

    @fact spring1e(3.0) --> [3.0 -3.0; -3.0 3.0]
    @fact spring1s(3.0, [3.0, 1.0]) --> 3.0 * (1.0 - 3.0)

end