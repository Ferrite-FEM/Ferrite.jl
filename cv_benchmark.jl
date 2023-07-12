using Ferrite, BenchmarkTools, StaticArrays

function get_values(CellType, ::Val{dim}, q_order=2) where dim 
    grid = generate_grid(CellType, ntuple(Returns(2), dim))
    ip = Ferrite.default_interpolation(getcelltype(grid))
    RefShape = Ferrite.getrefshape(ip)
    qr = QuadratureRule{RefShape}(q_order)
    cv_u = CellValues(qr, ip^dim, ip)
    cv_p = CellValues(qr, ip, ip)
    return cv_u, cv_p, getcoordinates(grid, 1)
end

function reinit_masterfix!(cv::Ferrite.OldCellValues{<:Any, <:Any, <:Tensor, <:Tensor, T, Vec{dim,T}}, x::AbstractVector{Vec{dim,T}}) where {dim, T}
    n_geom_basefuncs = Ferrite.getngeobasefunctions(cv)
    n_func_basefuncs = Ferrite.getnbasefunctions(cv)
    length(x) == n_geom_basefuncs || Ferrite.throw_incompatible_coord_length(length(x), n_geom_basefuncs)

    @inbounds for (i, w) in pairs(getweights(cv.qr))
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

#for (CT, dim) in ((Triangle,2), (QuadraticTriangle,2), (Hexahedron,3), (Tetrahedron,3))    
for (CT, dim) in ((Triangle,2),)
    # 2 and 4 fields in 2D
    cv_u, cv_p, x = get_values(CT, Val(dim), 2)
    ocv_u = Ferrite.OldCellValues(cv_u)
    ocv_p = Ferrite.OldCellValues(cv_p)
    
    print("Scalar      : $CT in $(dim)D"); println()
    print("1 PR          : "); @btime reinit!($cv_p, $x);
    print("1 master      : "); @btime reinit!($ocv_p, $x);
    print("1 master (fix): "); @btime reinit_masterfix!($ocv_p, $x);
    
    print("Vector      : $CT in $(dim)D"); println()
    print("1 PR          : "); @btime reinit!($cv_u, $x);
    print("1 master      : "); @btime reinit!($ocv_u, $x);
    print("1 master (fix): "); @btime reinit_masterfix!($ocv_u, $x);
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