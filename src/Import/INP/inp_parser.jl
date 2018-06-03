struct InpContent{dim, TF, N, TI}
    node_coords::Vector{NTuple{dim,TF}}
    celltype::String
    cells::Vector{NTuple{N,TI}}
    nodesets::Dict{String,Vector{TI}}
    cellsets::Dict{String,Vector{TI}}
    E::TF
    mu::TF
    density::TF
    nodedbcs::Dict{String, Vector{Tuple{TI,TF}}}
    cloads::Dict{Int, Vector{TF}}
    facesets::Dict{String, Vector{Tuple{TI,TI}}}
    dloads::Dict{String, TF}
end

const stopping_pattern = r"^\*[^\*]"

function extract_inp(filepath_with_ext)
    file = open(filepath_with_ext, "r")
    
    local node_coords
    local celltype, cells, offset
    nodesets = Dict{String,Vector{Int}}()
    cellsets = Dict{String,Vector{Int}}()
    local E, mu
    nodedbcs = Dict{String, Vector{Tuple{Int,Float64}}}()
    cloads = Dict{Int, Vector{Float64}}()
    facesets = Dict{String, Vector{Tuple{Int,Int}}}()
    dloads = Dict{String, Float64}()
    density = 0. # Should extract from the file

    node_heading_pattern = r"\*Node\s*,\s*NSET\s*=\s*([^,]*)"
    cell_heading_pattern = r"\*Element\s*,\s*TYPE\s*=\s*([^,]*)\s*,\s*ELSET\s*=\s*([^,]*)"
    nodeset_heading_pattern = r"\*NSET\s*,\s*NSET\s*=\s*([^,]*)"
    cellset_heading_pattern = r"\*ELSET\s*,\s*ELSET\s*=\s*([^,]*)"
    material_heading_pattern = r"\*MATERIAL\s*,\s*NAME\s*=\s*([^\s]*)"
    boundary_heading_pattern = r"\*BOUNDARY"
    cload_heading_pattern = r"\*CLOAD"
    dload_heading_pattern = r"\*DLOAD"

    line = readline(file)
    while !eof(file)
        m = match(node_heading_pattern, line)
        if m != nothing && m[1] == "Nall"
            node_coords, line = extract_nodes(file)
            dim = length(node_coords[1])
            continue
        end
        m = match(cell_heading_pattern, line)
        if m != nothing
            celltype = String(m[1])
            cellsetname = String(m[2]) 
            cells, offset, line = extract_cells(file)
            cellsets[cellsetname] = collect(1:length(cells))
            continue
        end
        m = match(nodeset_heading_pattern, line)
        if m != nothing
            nodesetname = String(m[1])
            line = extract_set!(nodesets, nodesetname, file)
            continue
        end
        m = match(cellset_heading_pattern, line)
        if m != nothing
            cellsetname = String(m[1])
            line = extract_set!(cellsets, cellsetname, file, offset)
            continue
        end
        m = match(material_heading_pattern, line)
        if m != nothing
            material_name = String(m[1])
            E, mu, line = extract_material(file)
            continue
        end
        m = match(boundary_heading_pattern, line)
        if m != nothing
            line = extract_nodedbcs!(nodedbcs, file)
            continue
        end
        m = match(cload_heading_pattern, line)
        if m != nothing
            line = extract_cload!(cloads, file, Val{dim})
            continue
        end
        m = match(dload_heading_pattern, line)
        if m != nothing
            line = extract_dload!(dloads, facesets, file, Val{dim}, offset)
            continue
        end
        line = readline(file)
    end

    close(file)

    return InpContent(node_coords, celltype, cells, nodesets, cellsets, E, mu, density, nodedbcs, cloads, facesets, dloads)
end
