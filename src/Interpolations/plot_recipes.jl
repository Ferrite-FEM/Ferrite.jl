@userplot DegreesOfFreedom

@recipe function plot(dof::DegreesOfFreedom)
  ip = dof.args[1]
  ylims --> (-1, 1)
  xlims --> (-1.5, 1.5)
  legend --> false
  label --> false
  markershape --> :circle
  markercolor --> :black
  markersize  --> 5

  @series begin
    seriestype = :scatter
    markershape := :none
    linspace(-1.0, 1.0, 5), zeros(5)
  end

  annots = []
  c = JuAFEM.get_dof_local_coordinates(ip)
  c_arr = zeros(1, length(c))
  for i in 1:length(c)
    push!(annots, (c[i][1], -0.1, i))
    c_arr[:, i] = c[i]
  end
  annotations := annots
  c_arr, zeros(c_arr)
end

@userplot ShapeFunction

function compute_values(ip::Lagrange{1, RefCube}, shape_func::Int)
  N_points = 50
  range = linspace(-1, 1, N_points)
  Ns = zeros(getnbasefunctions(ip))
  f(x) = JuAFEM.value!(ip, Ns, Vec{1}((x,)))[shape_func]
  return range, f
end

function compute_values(ip::Lagrange{2, RefCube}, shape_func::Int)
  N_points = 50
  range = linspace(-1, 1, N_points)
  Ns = zeros(getnbasefunctions(ip))
  f(x,y) = JuAFEM.value!(ip, Ns, Vec{2}((x,y)))[shape_func]
  return range, range, f
end

@recipe function plot(sf::ShapeFunction)
  ip, shape_func = sf.args[1], sf.args[2]
  if getdim(ip) == 1
    compute_values(ip, shape_func)
  elseif getdim(ip) == 2
    seriestype := :surface
    compute_values(ip, shape_func)
  end
end
