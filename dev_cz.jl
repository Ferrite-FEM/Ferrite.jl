using Pkg; Pkg.activate(".")
using Revise
using JuAFEM
using Test
using ForwardDiff

function test_cz_interp()

    dim_s = 2
    interpolation = CohesiveZone{1,RefCube,1,dim_s}()
    ndim = JuAFEM.getdim(interpolation)
    r_shape = JuAFEM.getrefshape(interpolation)
    func_order = JuAFEM.getorder(interpolation)
    @test typeof(interpolation) <: SurfaceInterpolation{ndim,r_shape,func_order,dim_s}

    n_basefuncs = getnbasefunctions(interpolation)
    x = rand(Tensor{1, ndim})
    f = (x) -> JuAFEM.value(interpolation, Tensor{1, ndim}(x))
    @test vec(ForwardDiff.jacobian(f, Array(x))') ≈
           reinterpret(Float64, JuAFEM.derivative(interpolation, x))
    @test sum(JuAFEM.value(interpolation, x)) ≈ 0.0
end

function test_surfacevectorvalues()
    x = [
        Vec{2,Float64}((0.0, 0.0)),
        Vec{2,Float64}((6.0, 0.0)),
        Vec{2,Float64}((0.0, 1.0)),
        Vec{2,Float64}((6.0, 1.0)),
        ]
    func_interpol = CohesiveZone{1,RefCube,1,2}()
    quad_rule = QuadratureRule{1,RefCube}(:lobatto,2)
    cv = SurfaceVectorValues(quad_rule, func_interpol)
    ndim = JuAFEM.getdim(func_interpol)
    n_basefuncs = getnbasefunctions(func_interpol)

    # fe_valtype == CellScalarValues && @test getnbasefunctions(cv) == n_basefuncs
    @test getnbasefunctions(cv) == n_basefuncs * JuAFEM.getspacedim(func_interpol)

    # x, n = valid_coordinates_and_normals(func_interpol)
    reinit!(cv, x)

    # Test computation of the jump vector
    u_vector = [2., 0., 3., 0.,
                4., 3., 4., 5.]

    val_qp1 = function_value(cv, 1, u_vector)
    @test val_qp1[1] ≈ 2.0
    @test val_qp1[2] ≈ 3.0
    val_qp2 = function_value(cv, 2, u_vector)
    @test val_qp2[1] ≈ 1.0
    @test val_qp2[2] ≈ 5.0

    # test integration
    area = 0.0
    for i in 1:getnquadpoints(cv)
        area += JuAFEM.getdetJdA(cv,i)
    end
    @test area ≈ 6.0

    @test spatial_coordinate(cv, 1, x) ≈ Vec{2}((0., 0.5))
    @test spatial_coordinate(cv, 2, x) ≈ Vec{2}((6., 0.5))

end

@testset "SurfaceVectorValues" begin
    test_surfacevectorvalues()
    test_cz_interp()
end

end


interp = CohesiveZone{1,RefCube,1,2}()
qr = QuadratureRule{1,RefCube}(2)
svv = SurfaceVectorValues(qr, interp)
_x = getcoordinates(grid, 1)
x = [_x[1], _x[2], _x[1], _x[2]]
reinit!(svv, x)
sum(svv.detJdA)

shape_value(svv, 1, 2)

x̄ = spatial_coordinate(svv, 1, x)




dh = DofHandler(grid)
push!(dh,:u, 2)
close!(dh)

dbc = ConstraintHandler(dh)
add!(dbc, Dirichlet(:u, getfaceset(dh.grid, "bottom"), (x,t) -> zero(Vec{2}), [1,2]))
add!(dbc, Dirichlet(:u, getfaceset(dh.grid, "top"), (x,t) -> Vec((0.0, x[1]*1.0)), [1,2]))
# add!(dbc, Dirichlet(:u, getfaceset(dh.grid, "bottom"), (x,t) -> 0.0, [1,2]))
# add!(dbc, Dirichlet(:u, getfaceset(dh.grid, "top"), (x,t) -> 1.0, [2]))
close!(dbc)
t = 0.0
update!(dbc, t)

n = ndofs(dh)
f = zeros(n)

K = create_sparsity_pattern(dh);
assembler = start_assemble(K, f)

n_basefuncs = getnbasefunctions(svv)

# ue = [2., 0., 2., 0.,
      # 4., 0., 6., 0.]
D = one(SymmetricTensor{2, 2})

a = zeros(n)
apply!(a, dbc)

Ke = zeros(n, n)
fe = zeros(n)
for cell in CellIterator(dh)
    dofs = celldofs(cell)
    ue = a[dofs]
    fill!(Ke, 0)
    fill!(fe, 0)
    for q_point in 1:getnquadpoints(svv)
        dΩ = JuAFEM.getdetJdA(svv, q_point)
        j = JuAFEM.function_value(svv, q_point, ue)
        T = D ⋅ j
        for i in 1:n_basefuncs
            v  = shape_value(svv, q_point, i)
            fe[i] += v ⋅ T * dΩ
            for j in 1:n_basefuncs
                u = shape_value(svv, q_point, j)
                Ke[i, j] += (v ⋅ D ⋅ u) * dΩ
            end
        end
    end
    assemble!(assembler, celldofs(cell), fe, Ke)
end

assembler.f

assembler.K
apply!(K, f, dbc)
u = K \ f

filename = "temp"
vtk_grid(filename, dh) do vtkfile
    vtk_point_data(vtkfile, dh, u)
end




ip = CohesiveZone{1,RefCube,1,2}()


G1 = svv.covar_base[1][:,1]
G2 = svv.covar_base[1][:,2]

# t = G1 × G2
# n = t/norm(t)
# H = t ⋅ n
#
svv.dMdξ
H = det(svv.covar_base[1])
Δc = 4-2
c = [2, 4]
Ba = svv.dNdξ[3,1]
Δc/2
surf_grad = norm(G1)/H * Ba

A = svv.covar_base[1][:,1]
B = svv.covar_base[1][:,2]
a = norm(B)/H * A/norm(A)
b = norm(A)/H * B/norm(B)

a⋅A
b⋅B



typeof(interp) <: SurfaceInterpolation{ndim,r_shape,func_order,dim_s}

# grad = a * ∂/∂ξ₁
grad = norm(B)/H
# grad = 1/svv.detJdA[1]

jump = JuAFEM.function_value(svv, 1, u2)


#
# struct CZ{dim,dim_s,shape,order} <: Interpolation{dim,shape,order} end
# JuAFEM.getnbasefunctions(::CZ{1,dim_s,RefCube,1}) where {dim_s} = 4
# JuAFEM.nvertexdofs(::CZ{1,dim_s,RefCube,1}) where {dim_s} = 1
# JuAFEM.reference_coordinates(::CZ{1,dim_s,RefCube,1})  where {dim_s}= JuAFEM.reference_coordinates(Lagrange{1,RefCube,1}())
#
#
# function jump_value(ip::CZ{1,dim_s,RefCube,1}, i::Int, ξ::Vec{1}) where {dim_s}
#     ξ_x = ξ[1]
#     i == 1 && return -(1 - ξ_x) * 0.5
#     i == 2 && return -(1 + ξ_x) * 0.5
#     i == 3 && return (1 - ξ_x) * 0.5
#     i == 4 && return (1 + ξ_x) * 0.5
#     throw(ArgumentError("no shape function $i for interpolation $ip"))
# end
#
# function mid_surf_value(ip::CZ{1,dim_s,RefCube,1}, i::Int, ξ::Vec{1}) where {dim_s}
#     ξ_x = ξ[1]
#     i == 1 && return (1 - ξ_x) * 0.5
#     i == 2 && return (1 + ξ_x) * 0.5
#     i == 4 && return (1 - ξ_x) * 0.5
#     i == 3 && return (1 + ξ_x) * 0.5
#     throw(ArgumentError("no shape function $i for interpolation $ip"))
# end
# in 2D these are the same
# g1 = G2 × n / H
# g2 = n × G1 / H
# g3 = n  # G1 × G2 / H


u2 = [2., 0., 2., 0.,
      4., 0., 4., 0.]

xx = 6.0

ϕ = 45
R = Tensor{2,2,Float64}((cosd(ϕ), -sind(ϕ), sind(ϕ), cos(ϕ)))
rot_corners = [c⋅R for c in corners]

corners = [Vec{2}((0.0, 0.0)),
         Vec{2}((xx, 0.0)),
         Vec{2}((xx, 0.0)),
         Vec{2}((0.0, 0.0))]

grid = generate_grid(JuAFEM.Quadrilateral, (3, 1), corners);
grid = generate_grid(JuAFEM.Quadrilateral, (1, 1), rot_corners);
x = getcoordinates(grid, 1)












N = [JuAFEM.value(interp, i, qr.points[1]) for i in 1:4]

function shape_jump(cv, q_point, i)
    n = getnbasefunctions(cv)
    if i <= n/2
        return -shape_value(cv, q_point, i)*2.0
    else
        return shape_value(cv, q_point, i)*2.0
    end
end
cvs = CellScalarValues(qr, interp, interp)
cvv = CellVectorValues(qr, interp, interp)



# Base.@pure _valuetype(::ScalarValues{dim}, ::AbstractVector{T}) where {dim,T} = T
# Base.@pure _valuetype(::ScalarValues{dim}, ::AbstractVector{Vec{dim,T}}) where {dim,T} = Vec{dim,T}
# Base.@pure _valuetype(cv::CellVectorValues{dim}, ::AbstractVector{T}) where {dim,T} = Vec{dim,T}

Base.@pure _valuetype(ip::Interpolation, ::AbstractVector{T}) where {T} = Vec{get_space_dim(interp),T}

get_space_dim(::Interpolation{dim}) where {dim} = dim
get_space_dim(::CZ{dim,dim_s}) where {dim,dim_s} = dim_s

get_space_dim(interp)


JuAFEM._valuetype(cvv, u1)

shape_value(cvv, 1, 4)
function_value(cvv, 1, u1)
u1 = [1., 2., 3., 4.]
u2 = [1., 2., 3., 4., 1., 2., 3., 4.]


# j = shape_jump(cv, 1, 1)
