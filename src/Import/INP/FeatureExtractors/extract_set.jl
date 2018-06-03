function extract_set!(sets::Dict{String, TV}, setname::AbstractString, file, offset=0) where {TI, TV<:AbstractVector{TI}}
    sets[setname] = Int[]
    vector = sets[setname]
    
    pattern_single = r"^(\d+)"
    pattern_subset = r"^([^,]+)"
    line = readline(file)
    m = match(stopping_pattern, line)
    while m isa Void
        m = match(pattern_single, line)
        if m != nothing
            push!(vector, parse(TI, m[1])-offset)
        else
            m = match(pattern_subset, line)
            if m != nothing
                subsetname = String(m[1])
                if haskey(sets, subsetname)
                    append!(vector, sets[subsetname])
                end
            end
        end
        line = readline(file)
        m = match(stopping_pattern, line)
    end
    return line
end
