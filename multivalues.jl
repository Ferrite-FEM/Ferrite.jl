using Ferrite, BenchmarkTools
import Ferrite: CellMultiValues

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
    cv, cv_u, cv_p, x = get_values(CT, Val(dim), 2)
    cv_u2 = deepcopy(cv_u); cv_p2 = deepcopy(cv_p)
    cv2 = CellMultiValues(a=cv_u, b=cv_p, c=cv_u2, d=cv_p2)

    println("$CT in $(dim)D")
    print("2 sep. vals: "); @btime reinit2!($cv_u, $cv_p, $x)
    print("2 multivals: "); @btime reinit!($cv, $x)
    print("4 sep. vals: "); @btime reinit4!($cv_u, $cv_p, $cv_u2, $cv_p2, $x)
    print("2 multivals: "); @btime reinit!($cv2, $x)
    println()
end
