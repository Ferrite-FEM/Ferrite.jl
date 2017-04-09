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

@recipe function plot(sf::ShapeFunction)
  ip, shape_func = sf.args[1], sf.args[2]
  N_points = 50
  range = linspace(-1, 1, N_points)
  range_vec = [Vec{1}((x,)) for x in range]
  Ns = zeros(getnbasefunctions(ip), N_points)
  for i in 1:length(range_vec)
      value!(ip, view(Ns, :, i), range_vec[i])
  end
  range, Ns[shape_func, :]
end
