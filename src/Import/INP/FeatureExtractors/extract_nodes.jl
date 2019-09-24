function extract_nodes(file, ::Type{TF}=Float64, ::Type{TI}=Int) where {TF, TI}
    line = readline(file)

    pattern = r"(\d+)\s*,\s*(-?\d+\.?\d*(e[-\+]?\d*)?)\s*,\s*(-?\d+\.?\d*(e[-\+]?\d*)?)\s*(,\s*(-?\d+\.?\d*(e[-\+]?\d*)?))?"
    m = match(pattern, line)
    
    first_node_idx = parse(TI, m[1])
    first_node_idx == TI(1) || throw("First node index is not 1.")

    if m[6] isa Void
        node_coords = [(parse(TF, m[2]), parse(TF, m[4]))]
        nextline = _extract_nodes!(node_coords, file, TI(1))
    else
        node_coords = [(parse(TF, m[2]), parse(TF, m[4]), parse(TF, m[7]))]
        nextline = _extract_nodes!(node_coords, file, TI(1))
    end
    return node_coords, nextline
end

function _extract_nodes!(node_coords::AbstractVector{NTuple{dim, TF}}, file, prev_node_idx::TI) where {dim, TF, TI}
    if dim === 2
        pattern = r"(\d+)\s*,\s*(-?\d+\.?\d*(e[-\+]?\d*)?)\s*,\s*(-?\d+\.?\d*(e[-\+]?\d*)?)"
    elseif dim === 3
        pattern = r"(\d+)\s*,\s*(-?\d+\.?\d*(e[-\+]?\d*)?)\s*,\s*(-?\d+\.?\d*(e[-\+]?\d*)?)\s*,\s*(-?\d+\.?\d*(e[-\+]?\d*)?)"
    else
        error("Dimension is not supported.")
    end

    line = readline(file)
    m = match(stopping_pattern, line)
    while m isa Void
        m = match(pattern, line)
        if m != nothing
            node_idx = parse(Int, m[1])
            node_idx == prev_node_idx + TI(1) || throw("Node indices are not consecutive.")
            if dim === 2
                push!(node_coords, (parse(TF, m[2]), parse(TF, m[4])))
            else
                push!(node_coords, (parse(TF, m[2]), parse(TF, m[4]), parse(TF, m[6])))
            end
            prev_node_idx = node_idx
        end
        line = readline(file)
        m = match(stopping_pattern, line)
    end
    return line
end
