"""
Purpose:
Generates a 2D rectangular mesh

Input:
p1    - Lower left point of rectangle [x1, y1]
p2    - Upper right point of rectangle [x2, y2]
nelx  - Number of elments in x direction
nely  - Number of elments in y direction
ndofs - Number of degrees of freedom per node

Output:
Edof  - Connectivity matrix for mesh, cf. Calfem Toolbox
Ex    - Elementwise x-coordinates, cf. Calfem Toolbox
Ey    - Elementwise y-coordinates, cf. Calfem Toolbox
Bi    - Matrix containing boundary dofs for segment i (i=1,2,3,4)
        First column -> 1st dofs, second column -> 2nd dofs and so on
        size = (num boundary nodes on segment) x ndofs
        B1 = Bottom side     B2 = Right side
        B3 = Upper side      B4 = Left side

Written by
Jim Brouzoulis
"""
function gen_quad_mesh(p1::Array, p2::Array, nelx::Int, nely::Int, ndofs::Int)

  nno = (nelx+1)*(nely+1) # number of nodes

  # First node on each 'row'
  a = collect(1:nelx+1:(nelx+1)*(nely+1))'-1

  #temp = collect(nelx+1:nelx+1:(nelx+1)*(nely))


  shift = nelx+1
  # Element connectivity - nodes for each element
  Elcon = int(zeros(nelx*nely, 4))
  el = 1
  for row = 1:nely
    for col = 1:nelx
      Elcon[el, :] = [col,col+1 ,col+shift+1, col + shift] + a[row]
      el +=1
    end
  end


  # Dofs for each node
  dofs = reshape( collect(1:(nelx+1)*(nely+1)*ndofs), ndofs,(nelx+1)*(nely+1))';

  # Element degrees of freedom
  Edof = int(zeros(nelx*nely, 1+4*ndofs))
  for el = 1:nely*nelx
    Edof[el,:] = [el dofs[Elcon[el,1],:] dofs[Elcon[el,2],:] dofs[Elcon[el,3],:] dofs[Elcon[el,4],:] ]
  end


  # Create element coordinates
  Y = linspace(p1[2],p2[2],nely+1)
  X = linspace(p1[1],p2[1],nelx+1)
  y = repmat(Y,1,nelx+1)'
  x = repmat(X,1,nely+1)
  coords = [x[:] y[:]]

  Ex = zeros(nelx*nely, 4)
  Ey = zeros(nelx*nely, 4)
  for el = 1:nely*nelx
    Ex[el,:] =  [coords[Elcon[el,1],1] coords[Elcon[el,2],1] coords[Elcon[el,3],1] coords[Elcon[el,4],1] ]
    Ey[el,:] =  [coords[Elcon[el,1],2] coords[Elcon[el,2],2] coords[Elcon[el,3],2] coords[Elcon[el,4],2] ]
  end

  # Boundary dofs
  B1 = dofs[1:nelx+1,:]
  B2 = dofs[(nelx+1)*(1:(nely+1)),:]
  B3 = dofs[(nelx+1)*(nely+1):-1:(nelx+1)*(nely)+1,:]
  B4 = dofs[(nelx+1)*(nely:-1:0)+1,:]

  return Edof, Ex, Ey, B1, B2, B3, B4, coords
end

