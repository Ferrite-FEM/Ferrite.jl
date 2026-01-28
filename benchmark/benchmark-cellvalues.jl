using Ferrite, BenchmarkTools

function element_routine(cv::CellValues)
    s = 0.0
    for q_point in 1:getnquadpoints(cv)
        for i in 1:getnbasefunctions(cv)
            for j in 1:getnbasefunctions(cv)
                s += norm(shape_value(cv, q_point, i)) * norm(shape_gradient(cv, q_point, j))
            end
        end
    end
    return s
end

function element_routine(cv::CellMultiValues)
    s = 0.0
    for q_point in 1:getnquadpoints(cv)
        for i in 1:getnbasefunctions(cv.s1)
            for j in 1:getnbasefunctions(cv.s1)
                s += norm(shape_value(cv.s1, q_point, i)) * norm(shape_gradient(cv.s1, q_point, j))
            end
        end
    end
    return s
end


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

# Check that the indexing inside the element routine doesn't affect the performance
for key in (:cv_s1, :cmv_s1_v1_s2_v2)
    cv = getproperty(Main, key)
    reinit!(cv, x)
    @btime element_routine($cv)
end
