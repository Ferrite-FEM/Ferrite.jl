using Revise, Ferrite, SparseArrays

function create_colored_cantilever_grid(celltype, n)
    grid = generate_grid(celltype, (10*n, n, n), Vec{3}((0.0, 0.0, 0.0)), Vec{3}((10.0, 1.0, 1.0)))
    colors = create_coloring(grid)
    return grid, colors
end;

function create_dofhandler(grid::Grid{dim}, ip) where {dim}
    dh = DofHandler(grid)
    add!(dh, :u, ip) # Add a displacement field
    close!(dh)
end;

function create_stiffness(::Val{dim}) where {dim}
    E = 200e9
    ν = 0.3
    λ = E*ν / ((1+ν) * (1 - 2ν))
    μ = E / (2(1+ν))
    δ(i,j) = i == j ? 1.0 : 0.0
    g(i,j,k,l) = λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))
    C = SymmetricTensor{4, dim}(g);
    return C
end;

# struct ScratchValues{T, CV <: CellValues, FV <: FaceValues, TT <: AbstractTensor, dim, Ti}
#     Ke::Matrix{T}
#     fe::Vector{T}
#     cellvalues::CV
#     facevalues::FV
#     global_dofs::Vector{Int}
#     ɛ::Vector{TT}
#     coordinates::Vector{Vec{dim, T}}
#     assembler::Ferrite.AssemblerSparsityPattern{T, Ti}
# end;

# function create_values(interpolation_space::Interpolation{refshape}, qr_order::Int) where {dim, refshape<:Ferrite.AbstractRefShape{dim}}
#     ## Interpolations and values
#     quadrature_rule = QuadratureRule{refshape}(qr_order)
#     face_quadrature_rule = FaceQuadratureRule{refshape}(qr_order)
#     cellvalues = [CellValues(quadrature_rule, interpolation_space) for i in 1:Threads.nthreads()];
#     facevalues = [FaceValues(face_quadrature_rule, interpolation_space) for i in 1:Threads.nthreads()];
#     return cellvalues, facevalues
# end;

# # Create a `ScratchValues` for each thread with the thread local data
# function create_scratchvalues(K, f, dh::DofHandler{dim}, ip) where {dim}
#     nthreads = Threads.nthreads()
#     assemblers = [start_assemble(K, f) for i in 1:nthreads]
#     cellvalues, facevalues = create_values(ip, 2)

#     n_basefuncs = getnbasefunctions(cellvalues[1])
#     global_dofs = [zeros(Int, ndofs_per_cell(dh)) for i in 1:nthreads]

#     fes = [zeros(n_basefuncs) for i in 1:nthreads] # Local force vector
#     Kes = [zeros(n_basefuncs, n_basefuncs) for i in 1:nthreads]

#     ɛs = [[zero(SymmetricTensor{2, dim}) for i in 1:n_basefuncs] for i in 1:nthreads]

#     coordinates = [[zero(Vec{dim}) for i in 1:length(dh.grid.cells[1].nodes)] for i in 1:nthreads]

#     return [ScratchValues(Kes[i], fes[i], cellvalues[i], facevalues[i], global_dofs[i],
#                          ɛs[i], coordinates[i], assemblers[i]) for i in 1:nthreads]
# end;

# # ## Threaded assemble

# # The assembly function loops over each color and does a threaded assembly for that color
# function doassemble(K::SparseMatrixCSC, colors, grid::Grid, dh::DofHandler, C::SymmetricTensor{4, dim}, ip) where {dim}

#     f = zeros(ndofs(dh))
#     scratches = create_scratchvalues(K, f, dh, ip)
#     b = Vec{3}((0.0, 0.0, 0.0)) # Body force

#     for color in colors
#         ## Each color is safe to assemble threaded
#         Threads.@threads :static for i in 1:length(color)
#             assemble_cell!(scratches[Threads.threadid()], color[i], K, grid, dh, C, b)
#         end
#     end

#     return K, f
# end

struct WorkStreamScratch{CV <: CellValues, S}
    cv::CV
    ε::Vector{S}
    C::SymmetricTensor{4, 3, Float64, 36}
end
function Base.copy(wss::WorkStreamScratch)
    return WorkStreamScratch(copy(wss.cv), copy(wss.ε), wss.C)
end

struct WorkStreamCopyData
    dofs::Vector{Int}
    Ke::Matrix{Float64}
    fe::Vector{Float64}
end
function Base.copy(wscd::WorkStreamCopyData)
    return WorkStreamCopyData(copy(wscd.dofs), copy(wscd.Ke), copy(wscd.fe))
end

function doassemble2(K::SparseMatrixCSC, colors, grid::Grid, dh::DofHandler, C::SymmetricTensor{4, dim}, ip::Interpolation{refshape}; chunk_size=2^8, ntasks=Threads.nthreads(), use_colors::Bool = true) where {dim, refshape}


    # Cell function
    cell_worker = function(cc::CellCache, scratch::WorkStreamScratch, data::WorkStreamCopyData)
        b = Vec{3}((0.0, 0.0, 0.0)) # Body force
        (; dofs, Ke, fe) = data
        local cellvalues = scratch.cv
        local ε = scratch.ε
        local C = scratch.C
        reinit!(cellvalues, cc)

        fill!(Ke, 0)
        fill!(fe, 0)

        n_basefuncs = getnbasefunctions(cellvalues)

        @inbounds for q_point in 1:getnquadpoints(cellvalues)
            for i in 1:n_basefuncs
                ε[i] = shape_symmetric_gradient(cellvalues, q_point, i)
            end
            dΩ = getdetJdV(cellvalues, q_point)
            for i in 1:n_basefuncs
                δu = shape_value(cellvalues, q_point, i)
                fe[i] += (δu ⋅ b) * dΩ
                ɛC = ε[i] ⊡ C
                for j in 1:n_basefuncs
                    εCε = εC ⊡ ε[j]
                    Ke[i, j] += ɛCɛ * dΩ
                end
            end
        end

        # Store the dofs
        copyto!(dofs, cc.dofs)
    end

    # Reducer
    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)

    copier = function(data)
        (; dofs, Ke, fe) = data
        for (i, I) in pairs(dofs)
            f[I] += fe[i]
            for (j, J) in pairs(dofs)
                K[I, J] += Ke[i, j]
            end
        end
        # assemble!(assembler, data.dofs, data.Ke, data.fe)
        return
    end

    qr_order = 2
    quadrature_rule = QuadratureRule{refshape}(qr_order)
    # face_quadrature_rule = FaceQuadratureRule{refshape}(qr_order)
    cellvalues = CellValues(quadrature_rule, ip)

    ɛ = [zero(SymmetricTensor{2, 3}) for i in 1:getnbasefunctions(cellvalues)]


    scratch_sample = WorkStreamScratch(cellvalues, ε, C)

    ndpc = ndofs_per_cell(dh)
    copy_sample = WorkStreamCopyData(zeros(Int, ndpc), zeros(ndpc, ndpc), zeros(ndpc))

    if use_colors
        # With colors
        Ferrite.mesh_loop(dh, colors, cell_worker, copier, scratch_sample, copy_sample; chunk_size = chunk_size, ntasks = ntasks)
    else
        # Without colors
        Ferrite.mesh_loop(dh,         cell_worker, copier, scratch_sample, copy_sample; chunk_size = chunk_size, ntasks = ntasks)
    end
    return K, f
end

# # The cell assembly function is written the same way as if it was a single threaded example.
# # The only difference is that we unpack the variables from our `scratch`.
# function assemble_cell!(scratch::ScratchValues, cell::Int, K::SparseMatrixCSC,
#                         grid::Grid, dh::DofHandler, C::SymmetricTensor{4, dim}, b::Vec{dim}) where {dim}

#     ## Unpack our stuff from the scratch
#     Ke, fe, cellvalues, facevalues, global_dofs, ɛ, coordinates, assembler =
#          scratch.Ke, scratch.fe, scratch.cellvalues, scratch.facevalues,
#          scratch.global_dofs, scratch.ɛ, scratch.coordinates, scratch.assembler

#     K = assembler.K
#     f = assembler.f

#     fill!(Ke, 0)
#     fill!(fe, 0)

#     n_basefuncs = getnbasefunctions(cellvalues)

#     ## Fill up the coordinates
#     nodeids = grid.cells[cell].nodes
#     for j in 1:length(coordinates)
#         coordinates[j] = grid.nodes[nodeids[j]].x
#     end

#     reinit!(cellvalues, coordinates)

#     for q_point in 1:getnquadpoints(cellvalues)
#         for i in 1:n_basefuncs
#             ɛ[i] = symmetric(shape_gradient(cellvalues, q_point, i))
#         end
#         dΩ = getdetJdV(cellvalues, q_point)
#         for i in 1:n_basefuncs
#             δu = shape_value(cellvalues, q_point, i)
#             fe[i] += (δu ⋅ b) * dΩ
#             ɛC = ɛ[i] ⊡ C
#             for j in 1:n_basefuncs
#                 Ke[i, j] += (ɛC ⊡ ɛ[j]) * dΩ
#             end
#         end
#     end

#     celldofs!(global_dofs, dh, cell)
#     for (i, I) in pairs(global_dofs)
#         f[I] += fe[i]
#         for (j, J) in pairs(global_dofs)
#             K[I, J] += Ke[i, j]
#         end
#     end
#     # assemble!(assembler, global_dofs, fe, Ke)
# end;

const n = 25
const grid, colors = create_colored_cantilever_grid(Hexahedron, n);
const ip = Lagrange{RefHexahedron,1}()^3
const dh = create_dofhandler(grid, ip);
const K = allocate_matrix(dh);
const C = create_stiffness(Val{3}());

## compilation
doassemble(K, colors, grid, dh, C, ip);
# @profview doassemble(K, colors, grid, dh, C, ip);
@elapsed @time doassemble(K, colors, grid, dh, C, ip);
@show sum(K.nzval)

# With copier
for i in 1:50
    @time doassemble2(K, colors, grid, dh, C, ip; use_colors = false)
end
@elapsed @time doassemble2(K, colors, grid, dh, C, ip; use_colors = true);
@show sum(K.nzval)
@elapsed @time doassemble2(K, colors, grid, dh, C, ip; use_colors = false);
@show sum(K.nzval)

# # Without copier
# doassemble3(K, colors, grid, dh, C, ip);
# @elapsed @time K, f = doassemble2(K, colors, grid, dh, C, ip);
# @show sum(K.nzval)

# run_assemble()

# Running the code with different number of threads give the following runtimes:
# * 1 thread  2.46 seconds
# * 2 threads 1.19 seconds
# * 3 threads 0.83 seconds
# * 4 threads 0.75 seconds

#md # ## [Plain program](@id threaded_assembly-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`threaded_assembly.jl`](threaded_assembly.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
