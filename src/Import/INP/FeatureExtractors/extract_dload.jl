function extract_dload!(dloads::Dict{String, TF}, facesets::Dict{String, Vector{Tuple{TI,TI}}}, file, ::Type{Val{dim}}, offset::TI) where {TI, TF, dim}
    pattern = r"(\d+)\s*,\s*P(\d+)\s*,\s*(\-?\d+\.\d*)"
    dload_heading_pattern = r"\*DLOAD"

    faceset_name = "DLOAD_SET_$(length(dloads)+1)"
    facesets[faceset_name] = Tuple{TI,TI}[]

    first = true
    prevload = zero(TF)
    local load
    
    line = readline(file)
    m = match(stopping_pattern, line)
    while m isa Void
        m = match(dload_heading_pattern, line)
        if m != nothing
            dloads[faceset_name] = load
            first = true
            faceset_name = "DLOAD_SET_$(length(dloads)+1)"
            facesets[faceset_name] = Tuple{TI,TI}[]
        end
        m = match(pattern, line)
        if m != nothing
            cellidx = parse(TI, m[1]) - offset
            faceidx = parse(TI, m[2])
            load = parse(TF, m[3])
            if !first && prevload != load
                throw("Loads in the same DLOAD set are not equal.")
            end
            prevload = load
            if first
                first = false
            end
            push!(facesets[faceset_name], (cellidx, faceidx))
        end
        line = readline(file)
        m = match(stopping_pattern, line)
    end
    dloads[faceset_name] = load

    return line
end
