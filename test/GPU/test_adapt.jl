function dofs_cpu(dh,cv)
    nbasefuncs = cv |> getnbasefunctions
    ncells = dh |> get_grid |> getncells
    dofs = zeros(Int32, nbasefuncs, ncells)
    for i in 1:ncells
        cdofs = celldofs(dh, i)
        dofs[:,i] .= cdofs
    end
    return dofs
end


function dofs_gpu_kernel(dofs, dh, cv)
    nbasefuncs = cv |> getnbasefunctions
    for cell in CellIterator(dh, convert(Int32,nbasefuncs))
        cdofs = celldofs(cell)
        dofs[:,cellid(cell)] .= cdofs
    end
    return nothing
end

weights_cpu(cv) = cv.qr |> getweights

function weights_gpu_kernel(weights, cv)
    nweights = length(weights)
    for i in 1:nweights
        weights[i] = cv.weights[i]
    end
end

function nodes_cpu(grid)
    nodes = grid.cells .|> (x -> x.nodes |> collect)
    return hcat(nodes...)
end

function nodes_gpu_kernel(nodes, dh, cv)
    nbasefuncs = cv |> getnbasefunctions
    for cell in CellIterator(dh, convert(Int32,nbasefuncs))
        cnodes = getnodes(cell)
        nodes[:,cellid(cell)] .= cnodes
    end
    return nothing
end

@testset "Adapt" begin
    dh, cv = generate_problem()
    cpudofs = dofs_cpu(dh,cv) |> cu
    ncells = dh |> get_grid |> getncells
    nbasefunctions = cv |> getnbasefunctions
    gpudofs = zeros(Int32, nbasefunctions, ncells) |> cu
    init_gpu_kernel(BackendCUDA, ncells, nbasefunctions, dofs_gpu_kernel, (gpudofs, dh, cv)) |> launch!
    ## Test that dofs are correctly transfered to the GPU
    @test all(cpudofs .== gpudofs)
    ## Test that weights are correctly transfered to the GPU
    cpuweights = weights_cpu(cv) |> cu
    gpuweights = zeros(Float32, length(cpuweights)) |> cu
    @cuda blocks = 1 threads = 1 weights_gpu_kernel(gpuweights, cv)
    @test all(cpuweights .== gpuweights)
    ## Test that nodes are correctly transfered to the GPU
    cpunodes = nodes_cpu(dh |> get_grid) |> cu
    n_nodes = length(cpunodes)
    gpu_cellnodes= CUDA.zeros(Int32,nbasefunctions,ncells)
    init_gpu_kernel(BackendCUDA, ncells, nbasefunctions, nodes_gpu_kernel, (gpu_cellnodes, dh, cv)) |> launch!
    @test all(cpunodes .== gpu_cellnodes)
end
