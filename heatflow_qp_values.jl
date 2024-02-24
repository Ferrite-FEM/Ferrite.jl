using Ferrite

# Standard element routine
function assemble_element_std!(Ke::Matrix, fe::Vector, cellvalues::CellValues)
    n_basefuncs = getnbasefunctions(cellvalues)
    ## Loop over quadrature points
    for q_point in 1:getnquadpoints(cellvalues)
        ## Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        ## Loop over test shape functions
        for i in 1:n_basefuncs
            δu  = shape_value(cellvalues, q_point, i)
            ∇δu = shape_gradient(cellvalues, q_point, i)
            ## Add contribution to fe
            fe[i] += δu * dΩ
            ## Loop over trial shape functions
            for j in 1:n_basefuncs
                ∇u = shape_gradient(cellvalues, q_point, j)
                ## Add contribution to Ke
                Ke[i, j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
    return Ke, fe
end

# Element routine using QuadratureValuesIterator
function assemble_element_qpiter!(Ke::Matrix, fe::Vector, cellvalues)
    n_basefuncs = getnbasefunctions(cellvalues)
    ## Loop over quadrature points
    for qv in Ferrite.QuadratureValuesIterator(cellvalues)
        ## Get the quadrature weight
        dΩ = getdetJdV(qv)
        ## Loop over test shape functions
        for i in 1:n_basefuncs
            δu  = shape_value(qv, i)
            ∇δu = shape_gradient(qv, i)
            ## Add contribution to fe
            fe[i] += δu * dΩ
            ## Loop over trial shape functions
            for j in 1:n_basefuncs
                ∇u = shape_gradient(qv, j)
                ## Add contribution to Ke
                Ke[i, j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
    return Ke, fe
end

function assemble_global(cellvalues, dh; kwargs...)
    assemble_global!(create_buffers(cellvalues, dh), cellvalues, dh; kwargs...)
end

function assemble_global!(buffer, cellvalues, dh::DofHandler; qp_iter::Val{QPiter}) where QPiter
    (;f, K, assembler, Ke, fe) = buffer
    for cell in CellIterator(dh)
        reinit!(cellvalues, getcoordinates(cell))
        fill!(Ke, 0)
        fill!(fe, 0)
        if QPiter
            assemble_element_qpiter!(Ke, fe, cellvalues)
        else
            assemble_element_std!(Ke, fe, cellvalues)
        end
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    return K, f
end

function create_buffers(cellvalues, dh)
    f = zeros(ndofs(dh))
    K = create_sparsity_pattern(dh)
    assembler = start_assemble(K, f)
    ## Local quantities
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    return (;f, K, assembler, Ke, fe)
end

n = 50
grid = generate_grid(Quadrilateral, (n, n));
ip = Lagrange{RefQuadrilateral, 1}()
qr = QuadratureRule{RefQuadrilateral}(2)

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh);

cellvalues = CellValues(qr, ip);
static_cellvalues = Ferrite.StaticCellValues(cellvalues)

Kstd, fstd = assemble_global(cellvalues, dh; qp_iter=Val(false));
K_qp, f_qp = assemble_global(cellvalues, dh; qp_iter=Val(true));
K_static, f_static = assemble_global(static_cellvalues, dh; qp_iter=Val(true));

using Test
@testset "check outputs" begin
    @test K_qp ≈ Kstd
    @test f_qp ≈ fstd

    @test K_static ≈ Kstd
    @test f_static ≈ fstd
end

# Benchmarking 
using BenchmarkTools
if n ≤ 100
    print("Standard: ")
    @btime assemble_global!(buffer, $cellvalues, $dh; qp_iter=Val(false)) setup=(buffer=create_buffers(cellvalues, dh));
    print("Std qpoint: ")
    @btime assemble_global!(buffer, $cellvalues, $dh; qp_iter=Val(true)) setup=(buffer=create_buffers(cellvalues, dh));
    print("Static qpoint: ")
    @btime assemble_global!(buffer, $static_cellvalues, $dh; qp_iter=Val(true)) setup=(buffer=create_buffers(static_cellvalues, dh));
else
    buffer = create_buffers(cellvalues, dh)
    print("Standard: ")
    assemble_global!(buffer, cellvalues, dh; qp_iter=Val(false))
    @time assemble_global!(buffer, cellvalues, dh; qp_iter=Val(false))
    print("Std qpoint: ")
    assemble_global!(buffer, cellvalues, dh; qp_iter=Val(true))
    @time assemble_global!(buffer, cellvalues, dh; qp_iter=Val(true))
    print("Static qpoint: ")
    assemble_global!(buffer, static_cellvalues, dh; qp_iter=Val(true))
    @time assemble_global!(buffer, static_cellvalues, dh; qp_iter=Val(true))
end
nothing

