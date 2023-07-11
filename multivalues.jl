using Ferrite, BenchmarkTools, StaticArrays
import Ferrite: MultiCellValues, MultiCellValues2, SingleCellValues

function _reinit!(cv::CellValues, x::AbstractVector{Vec{dim,T}}) where {dim,T}
    n_geom_basefuncs = getngeobasefunctions(cv)
    n_func_basefuncs = getnbasefunctions(cv)
    #length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)
    checkbounds(Bool, x, 1:getngeobasefunctions(geo_values)) || throw_incompatible_coord_length(length(x), getngeobasefunctions(geo_values))
    @inbounds for (i, w) in pairs(getweights(cv.qr))
        w = cv.qr.weights[i]
        fecv_J = zero(Tensor{2,dim,T})
        for j in 1:n_geom_basefuncs
            fecv_J += x[j] ⊗ cv.dMdξ[j, i]
        end
        detJ = det(fecv_J)
        detJ > 0.0 || throw_detJ_not_pos(detJ)
        cv.detJdV[i] = detJ * w
        Jinv = inv(fecv_J)
        for j in 1:n_func_basefuncs
            # cv.dNdx[j, i] = cv.dNdξ[j, i] ⋅ Jinv
            cv.dNdx[j, i] = dothelper(cv.dNdξ[j, i], Jinv)
        end
    end
end

function reinit_master!(cv::CellValues{<:Any, N_t, dNdx_t, dNdξ_t}, x::AbstractVector{Vec{dim,T}}) where {
    dim, T, vdim,
    N_t    <: Union{Number,   Vec{dim},       SVector{vdim}     },
    dNdx_t <: Union{Vec{dim}, Tensor{2, dim}, SMatrix{vdim, dim}},
    dNdξ_t <: Union{Vec{dim}, Tensor{2, dim}, SMatrix{vdim, dim}},
}
    n_geom_basefuncs = Ferrite.getngeobasefunctions(cv)
    n_func_basefuncs = getnbasefunctions(cv)
    length(x) == n_geom_basefuncs || Ferrite.throw_incompatible_coord_length(length(x), n_geom_basefuncs)

    @inbounds for (i, w) in pairs(Ferrite.getweights(cv.qr))
        fecv_J = zero(Tensor{2,dim,T})
        for j in 1:n_geom_basefuncs
            fecv_J += x[j] ⊗ cv.dMdξ[j, i]
        end
        detJ = det(fecv_J)
        detJ > 0.0 || Ferrite.throw_detJ_not_pos(detJ)
        cv.detJdV[i] = detJ * w
        Jinv = inv(fecv_J)
        for j in 1:n_func_basefuncs
            # cv.dNdx[j, i] = cv.dNdξ[j, i] ⋅ Jinv
            cv.dNdx[j, i] = Ferrite.dothelper(cv.dNdξ[j, i], Jinv)
        end
    end
end

function get_values(CellType, ::Val{dim}, q_order=2) where dim 
    grid = generate_grid(CellType, ntuple(Returns(2), dim))
    ip = Ferrite.default_interpolation(getcelltype(grid))
    RefShape = Ferrite.getrefshape(ip)
    qr = QuadratureRule{RefShape}(q_order)
    cv_u = CellValues(qr, ip^dim, ip)
    cv_p = CellValues(qr, ip, ip)
    return cv_u, cv_p, getcoordinates(grid, 1)
end

reinit2!(cv1, cv2, x) = (reinit!(cv1, x); reinit!(cv2, x))
reinit4!(cv1, cv2, cv3, cv4, x) = (reinit!(cv1, x); reinit!(cv2, x); reinit!(cv3, x); reinit!(cv4, x))

#for (CT, dim) in ((Triangle,2), (QuadraticTriangle,2), (Hexahedron,3), (Tetrahedron,3))    
for (CT, dim) in ((Triangle,2),)
    # 2 and 4 fields in 2D
    cv_u, cv_p, x = get_values(CT, Val(dim), 2)
    mcv_p = MultiCellValues(;a=cv_p)
    mcv2_p = MultiCellValues2(;a=cv_p)
    mcv_pu = MultiCellValues(;a=cv_p, b=cv_u)
    mcv2_pu = MultiCellValues2(;a=cv_p, b=cv_u)
    scv_p = SingleCellValues(cv_p)

    #cv_u2 = deepcopy(cv_u); cv_p2 = deepcopy(cv_p)
    #cv4 = MultiCellValues2(a=cv_u, b=cv_p, c=cv_u2, d=cv_p2)
    
    println("$CT in $(dim)D")
    print("1 CellValues        : "); @btime reinit!($cv_p, $x);
    print("1 CellValues(master): "); @btime reinit_master!($cv_p, $x);
    print("1 MultiCellValues   : "); @btime reinit!($mcv_p, $x);
    print("1 MultiCellValues2  : "); @btime reinit!($mcv2_p, $x);
    print("1 SingleCellValues  : "); @btime reinit!($scv_p, $x);
    # =#
    #=
    println()
    print("2 CellValues       : "); @btime reinit2!($cv_u, $cv_p, $x)
    print("2 MultiCellValues  : "); @btime reinit!($mcv_pu, $x)
    print("2 MultiCellValues2 : "); @btime reinit!($mcv2_pu, $x)
    println()
    # =#
    #=
    print("4 CellValues  : "); @btime reinit4!($cv_u, $cv_p, $cv_u2, $cv_p2, $x)
    print("4 MultiValues : "); @btime reinit!($cv4, $x)
    print("4 Tuple{CV}   : "); @btime reinit_multiple!($x, $cv_u, $cv_p, $cv_u2, $cv_p2)
    print("4 ValuesGroup : "); @btime reinit!($cvg4, $x);
    println()
    # =#
end
