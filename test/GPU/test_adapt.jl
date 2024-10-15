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


@testset "Adapt" begin
    dh, cv = generate_problem()
    cpudofs = dofs_cpu(dh,cv) |> cu
    ncells = dh |> get_grid |> getncells
    nbasefunctions = cv |> getnbasefunctions
    gpudofs = zeros(Int32, nbasefunctions, ncells) |> cu
    init_gpu_kernel(BackendCUDA, ncells, nbasefunctions, dofs_gpu_kernel, (gpudofs, dh, cv)) |> launch!
    @test all(cpudofs .== gpudofs)
end
