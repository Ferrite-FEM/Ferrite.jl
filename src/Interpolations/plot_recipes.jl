@recipe function plot{order}(ip::Lagrange{1, RefCube, order})
  ylims := (-1, 1)
  xlims := (-1.5, 1.5)
  legend := false
  label := false
  size = ()
  markershape := :circle
  markercolor := :black
  markersize  := 5

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
