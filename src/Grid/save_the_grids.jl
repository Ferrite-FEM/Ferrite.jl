function save_the_grid(grid::Grid,dir::String="/tmp",name::String="grid.toml")
	filename = dir * "/" * name 

	file = open(filename,"w")
	print(file,toml_saves_the_grids(grid))
	close(file)
end

export save_the_grid

function toml_saves_the_grids(grid::Grid,accuracy::Integer=8)
	t_string::String = ""

	t_string *= toml_saves_the_grids(grid.nodes)
	t_string *= toml_saves_the_grids(grid.nodesets,"Nodesets")
	t_string *= toml_saves_the_grids(grid.cellsets,"Cellsets")
	t_string *= toml_saves_the_grids(grid.vertexsets)
	t_string *= toml_saves_the_grids(grid.facetsets)
	t_string *= toml_saves_the_grids(grid.cells)

	return t_string
end

function toml_saves_the_grids(cells::AbstractVector{<:Ferrite.AbstractCell})
	t_string::String = ""
	for cell in cells
		t_string *= @sprintf("%s",toml_saves_the_grids(cell))
	end

	return t_string
end

function toml_saves_the_grids(cell::Ferrite.AbstractCell{rshape}) where {rshape}
	t_string::String = "[[Cells]]\n"
	t_string *= @sprintf("type = \"%s\"\n",typeof(cell))
	t_string *= @sprintf("nodes = %s\n",toml_saves_the_grids(cell.nodes))
	t_string *= "\n"

	return t_string
end

function toml_saves_the_grids(tpl::NTuple{N,<:Integer}) where N
	t_string::String = "[ "

	for n in 1:N
		t_string *= @sprintf("%d,",tpl[n])
	end

	t_string = t_string[1:end-1]

	t_string *= " ]"
end

function toml_saves_the_grids(facetsets::Dict{String,OrderedSet{FacetIndex}})
	t_string::String = "[Facetsets]\n"

	for (setname,facetset) in facetsets
		t_string *= @sprintf("%s = %s\n",setname,toml_saves_the_grids(facetset))
	end

	t_string *= "\n"

	return t_string
end

function toml_saves_the_grids(facetset::OrderedSet{FacetIndex})
	t_string::String = "[ "

	for facet in facetset
		t_string *= @sprintf("[%d,%d],",facet[1],facet[2])
	end

	t_string = t_string[1:end-1]

	t_string *= " ]"

	return t_string
end

# used for nodesets and cellsets
function toml_saves_the_grids(sets::Dict{String,OrderedSet{Int64}},collection_name::String)
	t_string::String = @sprintf("[%s]\n",collection_name)

	for (setname,cellset) in sets
		t_string *= @sprintf("%s = %s\n", setname, toml_saves_the_grids(cellset))
	end

	t_string *= "\n"

	return t_string
end

function toml_saves_the_grids(intset::OrderedSet{<:Integer})
	t_string::String = "[ "

	for x in intset
		t_string *= @sprintf("%d,",x)
	end

	t_string = t_string[1:end-1]

	t_string *= " ]"

	return t_string
end

function toml_saves_the_grids(nodes::AbstractVector{<:Node},accuracy::Integer=8)
	t_string::String = "[Nodes]\n"
	t_string *= "nodes = [ "

	for node in nodes
		t_string *= @sprintf("%s,",toml_saves_the_grids(node))
	end

	t_string = t_string[1:end-1]
	
	t_string *= " ]\n"
	t_string *= "\n"

	return t_string
end

function toml_saves_the_grids(node::Node{dim}) where dim
	t_string::String = "["

	for i in 1:dim
		t_string *= @sprintf("%s,",toml_saves_the_grids(node.x[i]))
	end

	t_string = t_string[1:end-1]

	t_string *= "]"

	return t_string
end

# find a small number of digits for precise storage
function toml_saves_the_grids(x::AbstractFloat)
	N = 17

	for n in 1:17
		if round(x,digits=n) - x == 0
			N = n

			break
		end
	end

	if N == 1
		t_string = @sprintf("%.0E",x)
	elseif N == 2
		t_string = @sprintf("%.1E",x)
	elseif N == 3
		t_string = @sprintf("%.2E",x)
	elseif N == 4
		t_string = @sprintf("%.3E",x)
	elseif N == 5
		t_string = @sprintf("%.4E",x)
	elseif N == 6
		t_string = @sprintf("%.5E",x)
	elseif N == 7
		t_string = @sprintf("%.6E",x)
	elseif N == 8
		t_string = @sprintf("%.7E",x)
	elseif N == 9
		t_string = @sprintf("%.8E",x)
	elseif N == 10
		t_string = @sprintf("%.9E",x)
	elseif N == 11
		t_string = @sprintf("%.10E",x)
	elseif N == 12
		t_string = @sprintf("%.11E",x)
	elseif N == 13
		t_string = @sprintf("%.12E",x)
	elseif N == 14
		t_string = @sprintf("%.13E",x)
	elseif N == 15
		t_string = @sprintf("%.14E",x)
	elseif N == 16
		t_string = @sprintf("%.15E",x)
	elseif N == 17
		t_string = @sprintf("%.16E",x)
	end

	return t_string
end

function toml_saves_the_grids(vertexsets::Dict{String,OrderedSet{VertexIndex}})
	t_string::String = "[Vertexsets]\n"

	for (setname,vertexset) in vertexsets
		t_string *= @sprintf("%s = %s\n",setname,toml_saves_the_grids(vertexset))
	end

	t_string *= "\n"

	return t_string
end

function toml_saves_the_grids(vertexset::OrderedSet{VertexIndex})
	t_string::String = "[ "

	for vertex in vertexset
		t_string *= @sprintf("[%d,%d],",vertex[1],vertex[2])
	end

	t_string = t_string[1:end-1]
	t_string *= " ]"

	return t_string
end

export toml_saves_the_grids
