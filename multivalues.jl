using Ferrite, BenchmarkTools
import Ferrite: CellMultiValues, CellSingleValues, CellValuesGroup
import Ferrite: reinit_multiple!

function get_values(CellType, ::Val{dim}, q_order=2) where dim 
    grid = generate_grid(CellType, ntuple(Returns(2), dim))
    ip = Ferrite.default_interpolation(getcelltype(grid))
    RefShape = Ferrite.getrefshape(ip)
    qr = QuadratureRule{dim,RefShape}(q_order)
    cv_u = CellVectorValues(qr, ip)
    cv_p = CellScalarValues(qr, ip)
    cv = CellMultiValues(;u=cv_u, p=cv_p)
    return cv, cv_u, cv_p, getcoordinates(grid, 1)
end

reinit2!(cv1, cv2, x) = (reinit!(cv1, x); reinit!(cv2, x))
reinit4!(cv1, cv2, cv3, cv4, x) = (reinit!(cv1, x); reinit!(cv2, x); reinit!(cv3, x); reinit!(cv4, x))

for (CT, dim) in ((Triangle,2), (QuadraticTriangle,2), (Hexahedron,3), (Tetrahedron,3))    
    # 2 and 4 fields in 2D
    cv2, cv_u, cv_p, x = get_values(CT, Val(dim), 2)
    csv_p = CellSingleValues(cv_p)
    cv1_p = CellMultiValues(;a=cv_p)
    cv_u2 = deepcopy(cv_u); cv_p2 = deepcopy(cv_p)
    cv4 = CellMultiValues(a=cv_u, b=cv_p, c=cv_u2, d=cv_p2)

    cvg1 = CellValuesGroup(;a=cv_p)
    cvg2 = CellValuesGroup(;a=cv_u, b=cv_p)
    cvg4 = CellValuesGroup(;a=cv_u, b=cv_p, c=cv_u2, d=cv_p2)

    println("$CT in $(dim)D")
    print("1 CellValues: : "); @btime reinit!($cv_p, $x);
    print("1 MultiValues : "); @btime reinit!($cv1_p, $x);
    print("1 SingleValues: "); @btime reinit!($csv_p, $x);
    print("1 Tuple{CV}   : "); @btime reinit_multiple!($x, $cv_p)
    print("1 ValuesGroup : "); @btime reinit!($cvg1, $x);
    println()
    print("2 CellValues  : "); @btime reinit2!($cv_u, $cv_p, $x)
    print("2 MultiValues : "); @btime reinit!($cv2, $x)
    print("2 Tuple{CV}   : "); @btime reinit_multiple!($x, $cv_u, $cv_p)
    print("2 ValuesGroup : "); @btime reinit!($cvg2, $x);
    println()
    print("4 CellValues  : "); @btime reinit4!($cv_u, $cv_p, $cv_u2, $cv_p2, $x)
    print("4 MultiValues : "); @btime reinit!($cv4, $x)
    print("4 Tuple{CV}   : "); @btime reinit_multiple!($x, $cv_u, $cv_p, $cv_u2, $cv_p2)
    print("4 ValuesGroup : "); @btime reinit!($cvg4, $x);
    println()
end
