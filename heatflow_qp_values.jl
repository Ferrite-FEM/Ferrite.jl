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

function assemble_element_qpiter!(Ke::Matrix, fe::Vector, cellvalues, cell_coords::AbstractVector)
    n_basefuncs = getnbasefunctions(cellvalues)
    ## Loop over quadrature points
    for qv in Ferrite.QuadratureValuesIterator(cellvalues, cell_coords)
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

function assemble_global!(buffer, cellvalues, dh::DofHandler; qp_iter::Val{QPiter}, reinit::Val{ReInit}) where {QPiter, ReInit}
    (;f, K, assembler, Ke, fe) = buffer
    for cell in CellIterator(dh)
        fill!(Ke, 0)
        fill!(fe, 0)
        if QPiter
            if ReInit
                reinit!(cellvalues, getcoordinates(cell))
                assemble_element_qpiter!(Ke, fe, cellvalues)
            else
                assemble_element_qpiter!(Ke, fe, cellvalues, getcoordinates(cell))
            end
        else
            reinit!(cellvalues, getcoordinates(cell))
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

stdassy(buffer, cv, dh) = assemble_global!(buffer, cv, dh; qp_iter=Val(false), reinit=Val(false))
qp_outside(buffer, cv, dh) = assemble_global!(buffer, cv, dh; qp_iter=Val(true), reinit=Val(true))
qp_inside(buffer, cv, dh) = assemble_global!(buffer, cv, dh; qp_iter=Val(true), reinit=Val(false))

Kstd, fstd = stdassy(create_buffers(cellvalues, dh), cellvalues, dh);
using LinearAlgebra
norm(Kstd)
K_qp_o, f_qp_o = qp_outside(create_buffers(cellvalues, dh), cellvalues, dh);
norm(K_qp_o)
K_qp_i, f_qp_i = qp_inside(create_buffers(cellvalues, dh), cellvalues, dh);
norm(K_qp_i)
cvs_o = Ferrite.StaticCellValues(cellvalues, Val(true)) # Save cell_coords in cvs_o
Ks_o, fs_o = qp_outside(create_buffers(cvs_o, dh), cvs_o, dh);
norm(Ks_o)
cvs_i = Ferrite.StaticCellValues(cellvalues, Val(false)) # Don't save cell_coords in cvs_o
Ks_i, fs_i = qp_inside(create_buffers(cvs_i, dh), cvs_i, dh);
norm(Ks_i)
using Test
@testset "check outputs" begin
    for (k, K, f) in (("qpo", K_qp_o, f_qp_o), ("qpi", K_qp_i, f_qp_i), ("so", Ks_o, fs_o), ("si", Ks_i, fs_i))
        @testset "$k" begin
          @test K ≈ Kstd
          @test f ≈ fstd
        end
    end
end

# Benchmarking
using BenchmarkTools
if n ≤ 100
    print("Standard: ")
    @btime stdassy(buffer, $cellvalues, $dh) setup=(buffer=create_buffers(cellvalues, dh));
    print("Std qpoint outside: ")
    @btime qp_outside(buffer, $cellvalues, $dh) setup=(buffer=create_buffers(cellvalues, dh));
    print("Std qpoint inside: ")
    @btime qp_inside(buffer, $cellvalues, $dh) setup=(buffer=create_buffers(cellvalues, dh));
    print("Static outside: ")
    @btime qp_outside(buffer, $cvs_o, $dh) setup=(buffer=create_buffers(cvs_o, dh));
    print("Static inside: ")
    @btime qp_inside(buffer, $cvs_i, $dh) setup=(buffer=create_buffers(cvs_i, dh));
else
    buffer = create_buffers(cellvalues, dh)
    print("Standard: ")
    @time stdassy(buffer, cellvalues, dh)
    print("Std qpoint outside: ")
    @time qp_outside(buffer, cellvalues, dh)
    print("Std qpoint inside: ")
    @time qp_inside(buffer, cellvalues, dh)
    print("Static outside: ")
    @time qp_outside(buffer, cvs_o, dh)
    print("Static inside: ")
    @time qp_inside(buffer, cvs_i, dh)
end
nothing
