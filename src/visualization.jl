function postprocess(solution)
   dim = length(solution[:,1]) 
   if dim == 1
        return reshape(solution, length(solution[1,:]))
   else 
        return reshape(sqrt.(sum(solution.^2, dims=1)), length(solution[1,:]))
    end 
end

function dof_to_node(dh::DofHandler, u::Array{T,1}; field::Int=1, process::Function = postprocess) where T
    fieldnames = JuAFEM.getfieldnames(dh)  
    field_dim = getfielddim(dh, field)
    data = fill(NaN, field_dim, getnnodes(dh.grid)) 
    offset = field_offset(dh, fieldnames[field])

    for cell in CellIterator(dh)
        _celldofs = celldofs(cell)
        counter = 1
        for node in cell.nodes
            for d in 1:field_dim
                data[d, node] = u[_celldofs[counter + offset]]
                counter += 1
            end
        end
    end
    return process(data)
end

to_triangle(::Union{Type{Triangle},Type{QuadraticTriangle}}, elements) = elements[:,1:3]
to_triangle(::Union{Type{Tetrahedron},Type{QuadraticTetrahedron}}, elements) = vcat(elements[:,[1,2,3]], elements[:,[1,2,4]], elements[:,[2,3,4]], elements[:,[1,4,3]])
to_triangle(::Union{Type{Quadrilateral},Type{QuadraticQuadrilateral}}, elements) = vcat(elements[:,[1,2,3]], elements[:,[3,4,1]])
to_triangle(::Union{Type{Hexahedron},Type{QuadraticHexahedron}}, elements) = vcat(elements[:,[1,2,3]], elements[:,[3,4,1]], elements[:,[1,5,6]], elements[:,[6,2,1]], elements[:,[2,6,7]], elements[:,[7,3,2]], elements[:,[3,7,8]], elements[:,[8,4,3]], elements[:,[1,4,8]], elements[:,[8,5,1]], elements[:,[5,8,7]], elements[:,[7,6,5]])

AbstractPlotting.plottype(::DofHandler{1,C,T}, ::Array{T,1}) where {C,T} = AbstractPlotting.lines
AbstractPlotting.plottype(::DofHandler{dim,C,T}, ::Array{T,1}) where {dim,C,T} = AbstractPlotting.mesh

function AbstractPlotting.plottype(grid::G) where G <: AbstractGrid
    if getdim(grid) == 1
        AbstractPlotting.scatterlines
    else
        AbstractPlotting.poly
    end
end

function AbstractPlotting.convert_arguments(::AbstractPlotting.PointBased, dh::DofHandler{1,C,T}, u::Array{T,1}) where {C,T}
    nodes = getnodes(dh.grid)
    coords = [node.x[1] for node in nodes]
    solution = dof_to_node(dh, u) 
    return ([AbstractPlotting.Point2f0(coords[i], solution[i]) for i in 1:getnnodes(dh.grid)],)
end

function AbstractPlotting.convert_arguments(::AbstractPlotting.PointBased, grid::Grid{1,C,T}) where {C,T}
    nodes = getnodes(grid)
    coords = [node.x[1] for node in nodes]
    return ([AbstractPlotting.Point2f0(coords[i], 0.0) for i in 1:getnnodes(grid)],)
end

function AbstractPlotting.mesh(dh::DofHandler, u::Array{T,1}, args...; process=postprocess, scale_plot=false, shading=false, kwargs...) where T
    C = getcelltype(dh.grid)
    nodes = getnodes(dh.grid)
    cells = getcells(dh.grid)
    coords = [node.x[i] for node in nodes, i in 1:getdim(dh.grid)]
    connectivity = getproperty.(cells, :nodes)
    N = length(vertices(cells[1]))
    elements = [element[i] for element in connectivity, i in 1:N]
    solution = dof_to_node(dh, u; process=process)
    triangle_elements = to_triangle(C, elements) 
    return AbstractPlotting.mesh(coords, triangle_elements, color=solution, args...; scale_plot=scale_plot, shading=shading, kwargs...)
end

function AbstractPlotting.surface(dh::DofHandler, u::Array{T,1}, args...; scale_plot=false, shading=false, kwargs...) where T
    @assert getdim(dh.grid) == 2 "Only 2D solutions supported!"
    C = getcelltype(dh.grid)
    nodes = getnodes(dh.grid)
    cells = getcells(dh.grid)
    coords = [node.x[i] for node in nodes, i in 1:2]
    connectivity = getproperty.(cells, :nodes)
    N = length(vertices(cells[1]))
    elements = [element[i] for element in connectivity, i in 1:N]
    solution = dof_to_node(dh, u)
    points = [AbstractPlotting.Point3f0(coord[1], coord[2], solution[idx]) for (idx, coord) in enumerate(eachrow(coords))]
    triangle_elements = to_triangle(C, elements)  
    return AbstractPlotting.mesh(points, triangle_elements, color=solution, args...; scale_plot=scale_plot, shading=shading, kwargs...)
end
