function generate_problem()
    left = Tensor{1,2,Float32}((0, -0))

    right = Tensor{1, 2, Float32}((rand(10.0:100000.0), rand(10.0:100000.0)))

    grid_dims = (rand(1:1000), rand(1:1000))

    grid = generate_grid(Quadrilateral, grid_dims, left, right)

    ip = Lagrange{RefQuadrilateral, 1}() # define the interpolation function (i.e. Bilinear lagrange)

    qr = QuadratureRule{RefQuadrilateral}(Float32,2)

    cellvalues = CellValues(Float32,qr, ip)

    dh = DofHandler(grid)

    add!(dh, :u, ip)

    close!(dh)

    return dh, cellvalues
end

function getalldofs(dh)
    ncells = dh |> get_grid |> getncells
    return map(i -> celldofs(dh, i) .|> Int32 , 1:ncells) |> (x -> hcat(x...)) |> cu
end


function dof_kernel_kernel!(dofs, dh,n_basefuncs)
    # this kernel is used to get all the dofs of the grid, which then
    # can be validated against the correct dofs (i.e. CPU version).
    for cell in CellIterator(dh, convert(Int32, n_basefuncs))
        cid = cellid(cell)
        cdofs = celldofs(cell)
        for i in 1:n_basefuncs
            dofs[Int32(i),cid] = cdofs[Int32(i)]
        end
    end
    return nothing
end


function assemble_element_gpu!(Ke,fe,cv,cell)
    n_basefuncs = getnbasefunctions(cv)
    for qv in Ferrite.QuadratureValuesIterator(cv,getcoordinates(cell))
        ## Get the quadrature weight
        dΩ = getdetJdV(qv)
        ## Loop over test shape functions
        for i in 1:n_basefuncs
            δu  = shape_value(qv, i)
            ∇u = shape_gradient(qv, i)
            ## Add contribution to fe
            fe[i] += δu * dΩ
            #fe_shared[tx,i] += δu * dΩ
            ## Loop over trial shape functions
            for j in 1:n_basefuncs
                ∇δu = shape_gradient(qv, j)
                ## Add contribution to Ke
                Ke[i,j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
end


function assemble_element_cpu!(Ke, fe, cellvalues)
    n_basefuncs = getnbasefunctions(cellvalues)
    # Loop over quadrature points
    for q_point in 1:getnquadpoints(cellvalues)
        # Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        # Loop over test shape functions
        for i in 1:n_basefuncs
            δu  = shape_value(cellvalues, q_point, i)
            ∇δu = shape_gradient(cellvalues, q_point, i)
            # Add contribution to fe
            fe[i] += δu * dΩ
            # Loop over trial shape functions
            for j in 1:n_basefuncs
                ∇u = shape_gradient(cellvalues, q_point, j)
                # Add contribution to Ke
                Ke[i, j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
    return Ke, fe
end

function localkefe_kernel!(kes,fes,cv,dh)
    nbasefuncs = getnbasefunctions(cv)
    for cell in CellIterator(dh, convert(Int32, nbasefuncs))
        Ke = cellke(cell)
        fe = cellfe(cell)
        assemble_element_gpu!(Ke, fe, cv, cell)
        kes[cellid(cell),:,:] .= Ke
        fes[cellid(cell),:] .= fe
    end
    return nothing
end

function get_cpu_kefe(dh, cellvalues)
    ncells = dh |> get_grid |> getncells
    n_basefuncs = getnbasefunctions(cellvalues)
    kes = zeros(Float32, ncells, n_basefuncs, n_basefuncs)
    fes = zeros(Float32, ncells, n_basefuncs)
    for cell in CellIterator(dh)
        cid = cellid(cell)
        reinit!(cellvalues, cell)
        # Compute element contribution
        assemble_element_cpu!((@view kes[cid,:,:]),(@view fes[cid,:,:]), cellvalues)
    end
    return kes |> cu , fes |> cu
end

@testset "Test iterators" begin
    dh, cellvalues  = generate_problem()
    n_basefuncs = getnbasefunctions(cellvalues)
    # 1. Test that dofs for each cell in the grid are correctly computed
    ncells = dh |> get_grid |> getncells
    dofs = CUDA.fill(Int32(0),n_basefuncs,ncells)
    correct_dofs = getalldofs(dh)
    kernel_config = CUDAKernelLauncher(ncells, n_basefuncs, dof_kernel_kernel!, (dofs, dh,n_basefuncs));
    launch_kernel!(kernel_config);
    @test all(dofs .≈ correct_dofs)

    # 2. Test that local ke and fe are correctly computed
    kes_gpu = CUDA.fill(0.0f0, ncells, n_basefuncs, n_basefuncs);
    fes_gpu = CUDA.fill(0.0f0, ncells, n_basefuncs);
    kernel_config = CUDAKernelLauncher(ncells, n_basefuncs, localkefe_kernel!, (kes_gpu, fes_gpu, cellvalues, dh));
    launch_kernel!(kernel_config);
    kes_cpu, fes_cpu = get_cpu_kefe(dh, cellvalues);
    @test all(abs.(kes_gpu .- kes_cpu) .< 1e-2) #TODO: This needs further investigation
    @test all(fes_gpu .≈ fes_cpu)
end
