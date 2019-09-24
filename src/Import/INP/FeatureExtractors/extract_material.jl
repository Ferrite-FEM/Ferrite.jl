function extract_material(file, ::Type{TF}=Float64) where TF
    elastic_heading_pattern = r"\*ELASTIC"
    Emu_pattern = r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)"
    line = readline(file)
    m = match(elastic_heading_pattern, line)
    if m != nothing
        line = readline(file)
        m = match(Emu_pattern, line)
        if m != nothing
            E = parse(TF, m[1])
            mu = parse(TF, m[2])
        end
    else
        throw("Material not supported.")
    end
    line = readline(file)
    return E, mu, line
end
