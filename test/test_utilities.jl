facts("assemble test") do
    K = zeros(3,3)
    Ke = [1 2 3; 4 5 6; 7 8 9]
    edof = [1  3 2 1]
    @fact assemble(edof, K, Ke) --> [9.0 8.0 7.0; 6.0 5.0 4.0; 3.0 2.0 1.0]

end

