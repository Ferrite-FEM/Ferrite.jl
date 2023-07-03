function visualize_grid(forest::ForestBWG{dim}) where dim
    fig = GLMakie.Figure()
    ax = GLMakie.Axis(fig[1,1])
    for tree in forest.cells
        for leaf in tree.leaves
            cellnodes = getnodes(forest,collect(tree.nodes)) .|> get_node_coordinate |> collect
            #request vertices and faces in octree coordinate system
            _vertices = Ferrite.vertices(leaf,tree.b)
            # transform from octree coordinate system to -1,1 by first shifting to 0,2 and later shift by -1
            _vertices = broadcast.(x->x .* 2/(2^tree.b) .- 1, _vertices) 
            octant_physical_coordinates = zeros(length(_vertices),dim)
            for (i,v) in enumerate(_vertices)
                octant_physical_coordinates[i,:] .= sum(j-> cellnodes[j] * Ferrite.shape_value(Lagrange{Ferrite.RefHypercube{dim},1}(),Vec{dim}(v),j),1:length(cellnodes)) 
            end
            GLMakie.scatter!(ax,octant_physical_coordinates,color=:black,markersize=25)
            center = sum(octant_physical_coordinates,dims=1) ./ 4
            #GLMakie.scatter!(ax,center,color=:black,markersize=25)
            facetable = dim == 2 ? Ferrite.𝒱₂ : Ferrite.𝒱₃
            for faceids in eachrow(facetable)                
                x = octant_physical_coordinates[faceids,1] + (octant_physical_coordinates[faceids,1] .- center[1])*0.02
                y = octant_physical_coordinates[faceids,2] + (octant_physical_coordinates[faceids,2] .- center[2])*0.02
                GLMakie.lines!(ax,x,y,color=:black)
            end
        end
    end
    return fig, ax
end
