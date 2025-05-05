function parse_the_grids(path::String="/tmp/grid.toml")
	file = open(path,"r")
	parsed_data = TOML.parse(file)
	close(file)

	cells = parse_the_cells(parsed_data)
	nodes = parse_the_nodes(parsed_data)
	cellsets = parse_the_cellsets(parsed_data)
	nodesets = parse_the_nodesets(parsed_data)
	facetsets = parse_the_facetsets(parsed_data)
	vertexsets = parse_the_vertexsets(parsed_data)

	return Grid(cells,nodes,cellsets,nodesets,facetsets,vertexsets)
end

export parse_the_grids

function parse_the_cells(parsed_data::Dict{String,Any})
	p_cells = parsed_data["Cells"]

	cells = AbstractCell[]

	for p_cell in p_cells
		p_nodes = p_cell["nodes"]
		p_type = p_cell["type"]

		cell_type = getfield(Ferrite,Symbol(p_type))
		N = length(p_nodes)

		cell = cell_type(NTuple{N,Int}(p_nodes)) 

		push!(cells,cell)
	end

	# restate the vector, so that julia finds an adequate type
	cells = [ cell for cell in cells ]

	return cells
end

function parse_the_nodes(parsed_data::Dict{String,Any})
	p_nodes = parsed_data["Nodes"]["nodes"]

	dim = length(p_nodes[1])

	nodes = Vector{Node{dim,Float64}}(undef,length(p_nodes))

	for (n,p_node) in enumerate(p_nodes)
		node_x = NTuple{dim,Float64}(p_node)
		node = Node(node_x)
		nodes[n] = node
	end

	return nodes
end

function parse_the_cellsets(parsed_data::Dict{String,Any})
	p_cellsets = parsed_data["Cellsets"]

	cellsets = Dict{String,OrderedSet{Int64}}()

	for (setname,p_cellset) in p_cellsets
		cellset = OrderedSet{Int64}(p_cellset)
		push!(cellsets, setname => cellset)
	end

	return cellsets
end

function parse_the_nodesets(parsed_data::Dict{String,Any})
	p_nodesets = parsed_data["Nodesets"]

	nodesets = Dict{String,OrderedSet{Int64}}()

	for (setname,p_nodeset) in p_nodesets
		nodeset = OrderedSet{Int64}(p_nodeset)
		push!(nodesets,setname => nodeset)
	end

	return nodesets
end

function parse_the_facetsets(parsed_data::Dict{String,Any})
	p_facetsets = parsed_data["Facetsets"]

	facetsets = Dict{String,OrderedSet{FacetIndex}}()

	for (setname,p_facetset) in p_facetsets
		facetset = OrderedSet{FacetIndex}()

		for p_facet in p_facetset
			facet = FacetIndex(p_facet[1],p_facet[2])
			push!(facetset,facet)
		end

		push!(facetsets,setname => facetset)
	end

	return facetsets
end

function parse_the_vertexsets(parsed_data::Dict{String,Any})
	p_vertexsets = parsed_data["Vertexsets"]

	vertexsets = Dict{String,OrderedSet{VertexIndex}}()

	for (setname,p_vertexset) in p_vertexsets
		vertexset = OrderedSet{VertexIndex}()

		for p_vertex in p_vertexset
			vertex = VertexIndex(p_vertex[1],p_vertex[2])
			push!(vertexset,vertex)
		end

		push!(vertexsets,setname => vertexset)
	end

	return vertexsets
end

