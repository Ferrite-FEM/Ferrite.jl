function extract_nodedbcs!(node_dbcs::Dict{String, Vector{Tuple{TI,TF}}}, file) where {TI, TF}
    pattern_zero = r"([^,]+)\s*,\s*(\d)"
    pattern_range = r"([^,]+)\s*,\s*(\d)\s*,\s*(\d)"
    pattern_other = r"([^,]+)\s*,\s*(\d)\s*,\s*(\d)\s*,\s*(\-?\d+\.\d*)"
    line = readline(file)
    m = match(stopping_pattern, line)
    while m isa Void
        m = match(pattern_other, line)
        if m != nothing
            nodesetname = m[1]
            dof1 = parse(TI, m[2])
            dof2 = parse(TI, m[3])
            val = parse(TF, m[4])
            if haskey(node_dbcs, nodesetname)
                for dof in dof1:dof2
                    push!(node_dbcs[nodesetname], (dof, val))
                end
            else
                node_dbcs[nodesetname] = [(dof1, val)]
                for dof in dof1+1:dof2
                    push!(node_dbcs[nodesetname], (dof, val))
                end
            end
            line = readline(file)
            m = match(stopping_pattern, line)
            continue
        end
        m = match(pattern_range, line)
        if m != nothing
            nodesetname = m[1]
            dof1 = parse(TI, m[2])
            dof2 = parse(TI, m[3])
            if haskey(node_dbcs, nodesetname)
                for dof in dof1:dof2
                    push!(node_dbcs[nodesetname], (dof, zero(TF)))
                end
            else
                node_dbcs[nodesetname] = [(dof1, zero(TF))]
                for dof in dof1+1:dof2
                    push!(node_dbcs[nodesetname], (dof, zero(TF)))
                end
            end
            line = readline(file)
            m = match(stopping_pattern, line)
            continue
        end
        m = match(pattern_zero, line)
        if m != nothing
            nodesetname = m[1]
            dof = parse(TI, m[2])
            if haskey(node_dbcs, nodesetname)
                push!(node_dbcs[nodesetname], (dof, zero(TF)))
            else
                node_dbcs[nodesetname] = [(dof, zero(TF))]
            end
            line = readline(file)
            m = match(stopping_pattern, line)
            continue
        end
        line = readline(file)
        m = match(stopping_pattern, line)
    end
    return line
end
