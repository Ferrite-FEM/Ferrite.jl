include("C:\\Users\\elias\\Dropbox\\Personliga saker\\Programmering\\Julia\\juafem_multiple_elements\\JuAFEM.jl\\src\\JuAFEM.jl")
#include("/Users/Elias/Dropbox/Personliga saker/Programmering/Julia/juafem_multiple_elements/JuAFEM.jl/src/JuAFEM.jl")
using JuAFEM

function doassemble{dim}(CELLTYPEID::Int, cellvalues::CellScalarValues{dim}, K::SparseMatrixCSC, assembler,  dh::DofHandler)
    b = 1.0
    #assembler = start_assemble(K, f)
    
    n_basefuncs = getnbasefunctions(cellvalues)
    global_dofs = zeros(Int, ndofs_per_cell(dh, CELLTYPEID))

    fe = zeros(n_basefuncs) # Local force vector
    Ke = zeros(n_basefuncs, n_basefuncs) # Local stiffness mastrix

    @inbounds for (cellcount, cell) in enumerate(CellIterator(dh, CELLTYPEID))
        fill!(Ke, 0)
        fill!(fe, 0)
        
        reinit!(cellvalues, cell)
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            for i in 1:n_basefuncs
                δT = shape_value(cellvalues, q_point, i)
                ∇δT = shape_gradient(cellvalues, q_point, i)
                fe[i] += (δT * b) * dΩ
                for j in 1:n_basefuncs
                    ∇T = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += (∇δT ⋅ ∇T) * dΩ
                end
            end
        end
        
        celldofs!(global_dofs, cell)
        assemble!(assembler, global_dofs, fe, Ke)
    end
    return K, f
end;

function creategrid()

    grid_quad = generate_grid(Quadrilateral, (10,10))
    grid_tria = generate_grid(Triangle, (10,10))

    nquadnodes = getnnodes(grid_quad)

    new_cells = Vector{Cell}()

    for cell in grid_tria.cellgroups[1]
        tmpnodes = Int[];
        for node in cell.nodes
            push!(tmpnodes,node+nquadnodes)
        end
        tmpcell = Triangle(NTuple(tuple(tmpnodes...)));
        push!(new_cells, tmpcell)
    end
    dim = 2
    new_nodes = Vector{Node{dim, Float64}}()
    b = Vec{2}((3.0, 0.0))
    for node in grid_tria.nodes
        push!(new_nodes, Node(node.x+b))
    end

    addcellgroup!(grid_quad, new_cells, new_nodes)
    return grid_quad
end

function create_small_grid()
    QUAD = Cell{dim,4,4}
    TRIA = Cell{dim,3,3}
    nodes = Vector{Node{dim,Float64}}()
    push!(nodes, Node((0.0,0.0)))
    push!(nodes, Node((1.0,0.0)))
    push!(nodes, Node((0.0,1.0)))
    push!(nodes, Node((1.0,1.0)))
    push!(nodes, Node((2.0,1.0)))
    cellgroup = Vector{Vector{Cell}}()
    quads = Vector{QUAD}()
    trias = Vector{TRIA}()
    push!(quads, QUAD((1,2,4,3)))
    push!(trias, TRIA((2,5,4)))
    push!(cellgroup,quads)
    push!(cellgroup,trias)

    grid = Grid(cellgroup,nodes)
end

dim = 2
QUAD = 1
TRIA = 2

grid = creategrid()
addnodeset!(grid, "boundary", x -> abs(x[1]) ≈ 1 ||  abs(x[2]) ≈ 1);

#Integration rule for quads
ip_quad = Lagrange{dim, RefCube, 1}()
qr_quad = QuadratureRule{dim, RefCube}(2)
cellvalues_quad = CellScalarValues(qr_quad, ip_quad);

#Integration rule for trias
ip_tria = Lagrange{dim, RefTetrahedron, 1}()
qr_tria = QuadratureRule{dim, RefTetrahedron}(2)
cellvalues_tria = CellScalarValues(qr_tria, ip_tria);

#Create dofhandler and add temperature field to all cellgroups
dh = DofHandler(grid,2)
push!(dh, :T, 1) 
close!(dh)

dbc = DirichletBoundaryConditions(dh)
add!(dbc, :T, getnodeset(grid, "boundary"), (x,t) -> 0.0 )
close!(dbc)
update!(dbc, 0.0)


K = create_sparsity_pattern(dh);
f = zeros(ndofs(dh))

#Asseble to stiffness matrix
assembler = start_assemble(K, f)
doassemble(QUAD, cellvalues_quad, K, assembler, dh)
doassemble(TRIA, cellvalues_tria, K, assembler, dh)

#Solve
apply!(K, f, dbc)
T = K \ f;

#Save
vtkfile = vtk_grid("plzwrk", dh)
vtk_point_data(vtkfile, dh, T)
vtk_save(vtkfile);