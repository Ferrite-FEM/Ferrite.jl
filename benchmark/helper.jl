# Shared functionality for the benchmark suite: batched sweeps over per-cell operations
# (a single `reinit!` or `celldofs!` is far below the ~1 us noise floor, so they are always
# measured over a batch of cells) and minimal but realistic assembly loops and element
# kernels, mirroring how the documented tutorials write them.
#
# Everything lives in a module so that including the benchmark files into `Main` (as
# PkgBenchmark and Tachometer do) does not leak helper names.
module FerriteBenchmarkHelpers

using Ferrite
using LinearAlgebra: norm

# Coordinates for `n` cells, cycling through the grid if it has fewer.
function cell_coordinate_batch(grid, n)
    return [getcoordinates(grid, mod1(i, getncells(grid))) for i in 1:n]
end

# Element dof vectors for a batch of cells (for scatter-only assembly benchmarks).
function celldofs_batch(dh, n)
    return [celldofs(dh, mod1(i, getncells(Ferrite.get_grid(dh)))) for i in 1:n]
end

#----------------------------------------------------------------------#
# Batched per-cell operations
#----------------------------------------------------------------------#

function reinit_sweep!(cv::CellValues, coord_batch)
    for x in coord_batch
        reinit!(cv, x)
    end
    return cv
end

# Non-identity mappings (e.g. Raviart-Thomas) require the cell for edge/face orientation.
function reinit_sweep!(cv::CellValues, cell_batch, coord_batch)
    for (cell, x) in zip(cell_batch, coord_batch)
        reinit!(cv, cell, x)
    end
    return cv
end

function reinit_sweep!(cmv::MultiFieldCellValues, coord_batch)
    for x in coord_batch
        reinit!(cmv, x)
    end
    return cmv
end

function reinit_sweep!(fv::FacetValues, coord_batch, nfacets_per_cell::Int)
    for (i, x) in pairs(coord_batch)
        reinit!(fv, x, mod1(i, nfacets_per_cell)) # cycle the local facet index
    end
    return fv
end

function celldofs_sweep!(dofs, dh)
    for i in 1:getncells(Ferrite.get_grid(dh))
        celldofs!(dofs, dh, i)
    end
    return dofs
end

# The inner loop of a typical residual evaluation: interpolate the solution and its
# gradient at every quadrature point. `reinit!` is measured separately, so it is done once
# and the quadrature loop repeats over a batch of element dof value vectors.
function function_values_sweep(cv, ue_batch)
    acc = 0.0
    for ue in ue_batch
        for q_point in 1:getnquadpoints(cv)
            acc += norm(function_value(cv, q_point, ue))
            acc += norm(function_gradient(cv, q_point, ue))
        end
    end
    return acc
end

# Query and traverse the neighborhood of every cell. The neighbor lists have to actually be
# iterated: the patches are precomputed, so just asking for their `length` is O(1) and gets
# optimized to nothing.
# spatial_coordinate at every quadrature point, e.g. for evaluating source terms.
function spatial_coordinate_sweep(cv, coord_batch)
    acc = 0.0
    for x in coord_batch
        for q_point in 1:getnquadpoints(cv)
            acc += norm(spatial_coordinate(cv, q_point, x))
        end
    end
    return acc
end

function neighborhood_sweep(top, grid)
    acc = 0
    for i in 1:getncells(grid)
        for neighbor in getneighborhood(top, grid, CellIndex(i))
            acc += neighbor
        end
    end
    return acc
end

# Iterate every vertex/edge/face neighbor list stored in the topology, i.e. a sweep over
# the raw adjacency storage (`ArrayOfVectorViews`), complementing `neighborhood_sweep`
# which goes through the `getneighborhood` query path (see issue #1019).
function neighbor_index_sum(neighbors::AbstractMatrix)
    acc = 0
    for nbs in neighbors
        for n in nbs
            acc += n[1] + n[2]
        end
    end
    return acc
end

function neighbor_index_sum(top::ExclusiveTopology)
    acc = neighbor_index_sum(top.vertex_vertex_neighbor)
    acc += neighbor_index_sum(top.edge_edge_neighbor)
    acc += neighbor_index_sum(top.face_face_neighbor)
    return acc
end

# getneighborhood for every facet of the (precomputed) facet skeleton: the FacetIndex
# query path, as used e.g. when assembling over interior facets (see issue #1019).
function skeleton_neighborhood_sweep(top, grid, skeleton)
    acc = 0
    for facetidx in skeleton
        acc += sum(n[1] + n[2] for n in getneighborhood(top, grid, facetidx); init = 0)
    end
    return acc
end

# A single CellValues construction is below the floor, so construct several.
function construction_sweep(qr, ip, ip_geo, n)
    local cv
    for _ in 1:n
        cv = CellValues(qr, ip, ip_geo)
    end
    return cv
end

# apply_zero! writes zeros to the constrained entries regardless of their current value, so
# repeated application to the same vector does identical work and needs no fresh setup.
function apply_zero_sweep!(f, ch, n)
    for _ in 1:n
        apply_zero!(f, ch)
    end
    return f
end

# update! re-evaluates the inhomogeneities; once per benchmarked "timestep" batch.
function update_sweep!(ch, n)
    for i in 1:n
        update!(ch, i / n)
    end
    return ch
end

# Like apply_zero_sweep!: apply_rhs! overwrites the constrained entries, so repeated
# application to the same vector does identical work.
function apply_rhs_sweep!(data, f, ch, n)
    for _ in 1:n
        apply_rhs!(data, f, ch)
    end
    return f
end

#----------------------------------------------------------------------#
# Element kernels
#----------------------------------------------------------------------#

function mass_kernel!(Ke, cv)
    fill!(Ke, 0)
    for q_point in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q_point)
        for i in 1:getnbasefunctions(cv)
            φᵢ = shape_value(cv, q_point, i)
            for j in 1:getnbasefunctions(cv)
                φⱼ = shape_value(cv, q_point, j)
                Ke[i, j] += (φᵢ * φⱼ) * dΩ
            end
        end
    end
    return Ke
end

function laplace_kernel!(Ke, cv)
    fill!(Ke, 0)
    for q_point in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q_point)
        for i in 1:getnbasefunctions(cv)
            ∇φᵢ = shape_gradient(cv, q_point, i)
            for j in 1:getnbasefunctions(cv)
                ∇φⱼ = shape_gradient(cv, q_point, j)
                Ke[i, j] += (∇φᵢ ⋅ ∇φⱼ) * dΩ
            end
        end
    end
    return Ke
end

function elasticity_kernel!(Ke, cv, C)
    fill!(Ke, 0)
    for q_point in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q_point)
        for i in 1:getnbasefunctions(cv)
            εᵢ = shape_symmetric_gradient(cv, q_point, i)
            for j in 1:getnbasefunctions(cv)
                εⱼ = shape_symmetric_gradient(cv, q_point, j)
                Ke[i, j] += (εᵢ ⊡ C ⊡ εⱼ) * dΩ
            end
        end
    end
    return Ke
end

# The divergence coupling block of a mixed u-p (Stokes-like) formulation, exercising
# shape_divergence and per-field access into a MultiFieldCellValues.
function mixed_divergence_kernel!(Ke, cmv)
    fill!(Ke, 0)
    for q_point in 1:getnquadpoints(cmv)
        dΩ = getdetJdV(cmv, q_point)
        for i in 1:getnbasefunctions(cmv.u)
            div_φ = shape_divergence(cmv.u, q_point, i)
            for j in 1:getnbasefunctions(cmv.p)
                ψ = shape_value(cmv.p, q_point, j)
                Ke[i, j] += (div_φ * ψ) * dΩ
            end
        end
    end
    return Ke
end

function elasticity_tensor(dim)
    E, ν = 200.0e9, 0.3
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2 * (1 + ν))
    δ(i, j) = i == j ? 1.0 : 0.0
    return SymmetricTensor{4, dim}((i, j, k, l) -> λ * δ(i, j) * δ(k, l) + μ * (δ(i, k) * δ(j, l) + δ(i, l) * δ(j, k)))
end

# Element kernel over a batch of cells, including `reinit!`, i.e. the per-cell work of an
# assembly loop without the iterator and scatter parts.
function element_sweep!(Ke, cv, coord_batch, kernel!::F, args...) where {F}
    for x in coord_batch
        reinit!(cv, x)
        kernel!(Ke, cv, args...)
    end
    return Ke
end

#----------------------------------------------------------------------#
# Global assembly loops
#----------------------------------------------------------------------#

# Full matrix + vector assembly: CellIterator, reinit!, element kernel and scatter.
function assemble_global!(K, f, dh, cv, kernel!::F, args...) where {F}
    Ke = zeros(ndofs_per_cell(dh), ndofs_per_cell(dh))
    fe = zeros(ndofs_per_cell(dh))
    assembler = start_assemble(K, f)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        kernel!(Ke, cv, args...)
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    return K
end

# Scatter-only loop with a precomputed element matrix, isolating `assemble!` into the
# sparse matrix from the kernel cost above.
function assemble_scatter!(K, dofs_batch, Ke)
    assembler = start_assemble(K)
    for dofs in dofs_batch
        assemble!(assembler, dofs, Ke)
    end
    return K
end

# Boundary load assembly (a pressure load p n ⋅ φ) over a facet set, for a vector field.
function assemble_facet_load!(f, dh, fv, facetset)
    fe = zeros(ndofs_per_cell(dh))
    fill!(f, 0)
    for facet in FacetIterator(dh, facetset)
        reinit!(fv, facet)
        fill!(fe, 0)
        for q_point in 1:getnquadpoints(fv)
            dΓ = getdetJdV(fv, q_point)
            n = getnormal(fv, q_point)
            for i in 1:getnbasefunctions(fv)
                fe[i] += (shape_value(fv, q_point, i) ⋅ n) * dΓ
            end
        end
        assemble!(f, celldofs(facet), fe)
    end
    return f
end

# DG interface term (interior penalty) over all interfaces: InterfaceIterator,
# InterfaceValues reinit! and scatter of the interface matrix.
function assemble_interfaces!(K, dh, iv, topology)
    n = getnbasefunctions(iv)
    Ki = zeros(n, n)
    assembler = start_assemble(K)
    for ic in InterfaceIterator(dh, topology)
        reinit!(iv, ic)
        fill!(Ki, 0)
        for q_point in 1:getnquadpoints(iv)
            dΓ = getdetJdV(iv, q_point)
            normal = getnormal(iv, q_point)
            for i in 1:n
                δu_jump = shape_value_jump(iv, q_point, i) * normal
                for j in 1:n
                    u_jump = shape_value_jump(iv, q_point, j) * normal
                    Ki[i, j] += (δu_jump ⋅ u_jump) * dΓ
                end
            end
        end
        assemble!(assembler, interfacedofs(ic), Ki)
    end
    return K
end

end # module FerriteBenchmarkHelpers
