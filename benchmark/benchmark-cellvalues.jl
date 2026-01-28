using Ferrite, BenchmarkTools

CT = QuadraticHexahedron
ip_geo = geometric_interpolation(CT)
x = Ferrite.reference_coordinates(ip_geo)
dim = Ferrite.getrefdim(ip_geo)
RefShape = getrefshape(ip_geo)
qr = QuadratureRule{RefShape}(2)
ip1 = Lagrange{RefShape, 1}()
ip2 = Lagrange{RefShape, 2}()

update_gradients = true
update_hessians = false

cv_s1 = CellValues(qr, ip1, ip_geo; update_gradients, update_hessians)
cv_v1 = CellValues(qr, ip1^dim, ip_geo; update_gradients, update_hessians)
cv_s2 = CellValues(qr, ip2, ip_geo; update_gradients, update_hessians)
cv_v2 = CellValues(qr, ip2^dim, ip_geo; update_gradients, update_hessians)

cmv_s1 = CellMultiValues(qr, (s1 = ip1,), ip_geo; update_gradients, update_hessians)
cmv_s1_square = CellMultiValues(qr, (s1 = ip1, s2 = ip1), ip_geo; update_gradients, update_hessians)
cmv_s1_v1 = CellMultiValues(qr, (s1 = ip1, v1 = ip1^dim), ip_geo; update_gradients, update_hessians)
cmv_s2_v2 = CellMultiValues(qr, (s2 = ip2, v2 = ip2^dim), ip_geo; update_gradients, update_hessians)
cmv_s1_v1_s2_v2 = CellMultiValues(qr, (s1 = ip1, v1 = ip1^dim, s2 = ip2, v2 = ip2^dim), ip_geo; update_gradients, update_hessians)

for key in (
        :cv_s1, :cv_v1, :cv_s2, :cv_v2,
        :cmv_s1, :cmv_s1_square, :cmv_s1_v1, :cmv_s2_v2, :cmv_s1_v1_s2_v2,
    )
    cv = getproperty(Main, key)
    print(key, ": ")
    @btime reinit!($cv, $x)
end
