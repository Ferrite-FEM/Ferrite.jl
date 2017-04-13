
using JuAFEM
using Plots
pgfplots()

@userplot ShapeFunction





ip = Lagrange{2, RefCube, 1}()
plotlyjs()
shapefunction(ip, 3)


ip

p = plot(legend = true)
for i in 1:JuAFEM.getnbasefunctions(ip)
end
p




plot(range, Ns[3,:])



y = rand(10)
plot(y,annotations=(3,y[3],text("this is 3",:left)),leg=false)
