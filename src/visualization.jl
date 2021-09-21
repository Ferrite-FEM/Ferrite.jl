function postprocess(node_values)
    dim = length(node_values)
    if dim == 1
        return node_values
    else 
        return sqrt(sum(node_values.^2))
    end 
end

function dof_to_node(dh::AbstractDofHandler, u::Array{T,1}; field::Int=1, process::Function=postprocess) where T
    fieldnames = Ferrite.getfieldnames(dh)  
    field_dim = getfielddim(dh, field)
    data = fill(NaN, getnnodes(dh.grid), field_dim) 
    offset = field_offset(dh, fieldnames[field])

    for cell in CellIterator(dh)
        _celldofs = celldofs(cell)
        counter = 1
        for node in cell.nodes
            for d in 1:field_dim
                data[node, d] = u[_celldofs[counter + offset]]
                counter += 1
            end
        end
    end
    return mapslices(process, data, dims=[2])
end

to_triangle(::Type{<:AbstractCell{2,N,3}}, elements) where N = elements[:,1:3]
to_triangle(::Type{<:AbstractCell{3,N,4}}, elements) where N = vcat(elements[:,[1,2,3]], elements[:,[1,2,4]], elements[:,[2,3,4]], elements[:,[1,4,3]])
to_triangle(::Type{<:AbstractCell{2,N,4}}, elements) where N = vcat(elements[:,[1,2,3]], elements[:,[3,4,1]])
to_triangle(::Type{<:AbstractCell{3,N,6}}, elements) where N = vcat(elements[:,[1,2,3]], elements[:,[3,4,1]], elements[:,[1,5,6]], elements[:,[6,2,1]], elements[:,[2,6,7]], elements[:,[7,3,2]], elements[:,[3,7,8]], elements[:,[8,4,3]], elements[:,[1,4,8]], elements[:,[8,5,1]], elements[:,[5,8,7]], elements[:,[7,6,5]])

AbstractPlotting.plottype(::Union{DofHandler{1,C,T},MixedDofHandler{1,T}}, ::Array{T,1}) where {C,T} = AbstractPlotting.lines
AbstractPlotting.plottype(::Union{DofHandler{dim,C,T},MixedDofHandler{dim,T}}, ::Array{T,1}) where {dim,C,T} = AbstractPlotting.mesh

function AbstractPlotting.convert_arguments(::AbstractPlotting.PointBased, dh::Union{DofHandler{1,C,T},MixedDofHandler{1,T}}, u::Array{T,1}) where {C,T}
    nodes = getnodes(dh.grid)
    coords = [node.x[1] for node in nodes]
    solution = dof_to_node(dh, u) 
    return ([AbstractPlotting.Point2f0(coords[i], solution[i]) for i in 1:getnnodes(dh.grid)],)
end

function AbstractPlotting.mesh(dh::AbstractDofHandler, u::Array{T,1}, args...; field::Int=1, process::Function=postprocess, scale_plot=false, shading=false, kwargs...) where T
    C = getcelltype(dh.grid)
    nodes = getnodes(dh.grid)
    cells = getcells(dh.grid)
    coords = [node.x[i] for node in nodes, i in 1:getdim(dh.grid)]
    connectivity = getproperty.(cells, :nodes)
    N = length(vertices(cells[1]))
    elements = [element[i] for element in connectivity, i in 1:N]
    solution = reshape(dof_to_node(dh, u; field=field, process=process), getnnodes(dh.grid))
    triangle_elements = to_triangle(C, elements) 
    return AbstractPlotting.mesh(coords, triangle_elements, color=solution, args...; scale_plot=scale_plot, shading=shading, kwargs...)
end

function AbstractPlotting.mesh!(dh::AbstractDofHandler, u::Array{T,1}, args...; field::Int=1, process::Function=postprocess, scale_plot=false, shading=false, kwargs...) where T
    C = getcelltype(dh.grid)
    nodes = getnodes(dh.grid)
    cells = getcells(dh.grid)
    coords = [node.x[i] for node in nodes, i in 1:getdim(dh.grid)]
    connectivity = getproperty.(cells, :nodes)
    N = length(vertices(cells[1]))
    elements = [element[i] for element in connectivity, i in 1:N]
    solution = reshape(dof_to_node(dh, u; field=field, process=process), getnnodes(dh.grid))
    triangle_elements = to_triangle(C, elements) 
    return AbstractPlotting.mesh!(coords, triangle_elements, color=solution, args...; scale_plot=scale_plot, shading=shading, kwargs...)
end

function AbstractPlotting.surface(dh::AbstractDofHandler, u::Array{T,1}, args...; field::Int=1, process::Function=postprocess, scale_plot=false, shading=false, kwargs...) where T
    @assert getdim(dh.grid) == 2 "Only 2D solutions supported!"
    C = getcelltype(dh.grid)
    nodes = getnodes(dh.grid)
    cells = getcells(dh.grid)
    coords = [node.x[i] for node in nodes, i in 1:2]
    connectivity = getproperty.(cells, :nodes)
    N = length(vertices(cells[1]))
    elements = [element[i] for element in connectivity, i in 1:N]
    solution = reshape(dof_to_node(dh, u; field=field, process=process), getnnodes(dh.grid))
    points = [AbstractPlotting.Point3f0(coord[1], coord[2], solution[idx]) for (idx, coord) in enumerate(eachrow(coords))]
    triangle_elements = to_triangle(C, elements)  
    return AbstractPlotting.mesh(points, triangle_elements, color=solution, args...; scale_plot=scale_plot, shading=shading, kwargs...)
end

function AbstractPlotting.surface!(dh::AbstractDofHandler, u::Array{T,1}, args...; field::Int=1, process::Function=postprocess, scale_plot=false, shading=false, kwargs...) where T
    @assert getdim(dh.grid) == 2 "Only 2D solutions supported!"
    C = getcelltype(dh.grid)
    nodes = getnodes(dh.grid)
    cells = getcells(dh.grid)
    coords = [node.x[i] for node in nodes, i in 1:2]
    connectivity = getproperty.(cells, :nodes)
    N = length(vertices(cells[1]))
    elements = [element[i] for element in connectivity, i in 1:N]
    solution = reshape(dof_to_node(dh, u; field=field, process=process), getnnodes(dh.grid))
    points = [AbstractPlotting.Point3f0(coord[1], coord[2], solution[idx]) for (idx, coord) in enumerate(eachrow(coords))]
    triangle_elements = to_triangle(C, elements)  
    return AbstractPlotting.mesh!(points, triangle_elements, color=solution, args...; scale_plot=scale_plot, shading=shading, kwargs...)
end

function AbstractPlotting.arrows(dh::AbstractDofHandler, u::Array{T,1}, args...; field::Int=1, arrowsize=0.08, normalize=true, kwargs...) where T
    C = getcelltype(dh.grid)
    nodes = getnodes(dh.grid)
    cells = getcells(dh.grid)
    coords = [node.x[i] for node in nodes, i in 1:getdim(dh.grid)]
    connectivity = getproperty.(cells, :nodes)
    N = length(vertices(cells[1]))
    elements = [element[i] for element in connectivity, i in 1:N]
    solution = dof_to_node(dh, u; field=field, process=identity)
    triangle_elements = to_triangle(C, elements) 
    if getdim(dh.grid) == 2
        AbstractPlotting.arrows(coords[:,1], coords[:,2], solution[:,1], solution[:,2], args...; arrowsize=arrowsize, normalize=normalize, kwargs...)
    elseif getdim(dh.grid) == 3
        AbstractPlotting.arrows(coords[:,1], coords[:,2], coords[:,3], solution[:,1], solution[:,2], solution[:,3], args...; arrowsize=arrowsize, normalize=normalize, kwargs...)
    end
end

function AbstractPlotting.arrows!(dh::AbstractDofHandler, u::Array{T,1}, args...; field::Int=1, arrowsize=0.08, normalize=false, kwargs...) where T
    C = getcelltype(dh.grid)
    nodes = getnodes(dh.grid)
    cells = getcells(dh.grid)
    coords = [node.x[i] for node in nodes, i in 1:getdim(dh.grid)]
    connectivity = getproperty.(cells, :nodes)
    N = length(vertices(cells[1]))
    elements = [element[i] for element in connectivity, i in 1:N]
    solution = dof_to_node(dh, u; field=field, process=identity)
    triangle_elements = to_triangle(C, elements) 
    if getdim(dh.grid) == 2
        AbstractPlotting.arrows!(coords[:,1], coords[:,2], solution[:,1], solution[:,2], args...; arrowsize=arrowsize, normalize=normalize, kwargs...)
    elseif getdim(dh.grid) == 3
        AbstractPlotting.arrows!(coords[:,1], coords[:,2], coords[:,3], solution[:,1], solution[:,2], solution[:,3], args...; arrowsize=arrowsize, normalize=normalize, kwargs...)
    end
end

function warp_by_vector(dh::AbstractDofHandler, u::Array{T,1}, args...; field::Int=1, scale=1.0, process::Function=postprocess, scale_plot=false, shading=false, kwargs...) where T
    C = getcelltype(dh.grid)
    nodes = getnodes(dh.grid)
    cells = getcells(dh.grid)
    coords = [node.x[i] for node in nodes, i in 1:getdim(dh.grid)]
    connectivity = getproperty.(cells, :nodes)
    N = length(vertices(cells[1]))
    elements = [element[i] for element in connectivity, i in 1:N]
    u_matrix = dof_to_node(dh, u; field=field, process=identity)
    solution = reshape(dof_to_node(dh, u; field=field, process=process), getnnodes(dh.grid))
    triangle_elements = to_triangle(C, elements) 
    return AbstractPlotting.mesh(coords .+ (scale .* u_matrix), color=solution, triangle_elements, args...; scale_plot=scale_plot, shading=shading, kwargs...)
end
