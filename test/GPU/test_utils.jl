function generate_Bilinear_problem()
    left = Tensor{1, 2, Float32}((0, -0))

    right = Tensor{1, 2, Float32}((rand(10.0:100000.0), rand(10.0:100000.0)))

    grid_dims = (rand(1:1000), rand(1:1000))

    grid = generate_grid(Quadrilateral, grid_dims, left, right)

    ip = Lagrange{RefQuadrilateral, 1}() # define the interpolation function (i.e. Bilinear lagrange)

    qr = QuadratureRule{RefQuadrilateral}(Float32, 2)

    cellvalues = CellValues(Float32, qr, ip)

    dh = DofHandler(grid)

    add!(dh, :u, ip)

    close!(dh)

    return dh, cellvalues
end

function generate_Biquadratic_problem()
    left = Tensor{1, 2, Float32}((0, -0))

    right = Tensor{1, 2, Float32}((rand(10.0:100000.0), rand(10.0:100000.0)))

    grid_dims = (rand(100:1000), rand(100:1000)) # to make sure the problem is big enough to use `CUDAGlobalCellIterator`

    grid = generate_grid(Quadrilateral, grid_dims, left, right)

    ip = Lagrange{RefQuadrilateral, 2}() # define the interpolation function (i.e. Biquadratic lagrange)

    qr = QuadratureRule{RefQuadrilateral}(Float32, 3) # 3x3 quadrature rule

    cellvalues = CellValues(Float32, qr, ip)

    dh = DofHandler(grid)

    add!(dh, :u, ip)

    close!(dh)

    return dh, cellvalues
end
