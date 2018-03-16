function generate(file)
    dir = joinpath(@__DIR__, "src", "examples")
    name = first(splitext(last(splitdir(file))))
    md = julia_example_to_markdown(file)
    # replace some things
    md = replace(md, "@__NAME__" => name)
    md = replace(md, "@__DIR__" => dir)
    code = """
        ```@setup $(name)
        include("$(@__FILE__)")
        write("$(name).jl", extract_code("$(file)"))
        ```
        ```julia
        $(extract_code(file))
        ```
        """
    md = replace(md, "@__CODE__" => code)
    open(joinpath(dir, name*".md"), "w") do f
        write(f, md)
        write(f,
            """

            *This page was rendered automatically based on
            [`$(name).jl`](https://github.com/KristofferC/JuAFEM.jl/blob/docs/examples/$(name).jl)*
            """)
    end
    return "examples/$(name).md"
end

# strip all comments
function extract_code(file)
    io = IOBuffer() # output
    for line in eachline(file, chomp = false)
        (!startswith(strip(line), "#") || isempty(strip(line))) && write(io, line)
    end
    strip(String(take!(io)))
end

function julia_example_to_markdown(file)
    io = IOBuffer()
    for line in eachline(file)
        if isempty(line)
            write(io, '\n')
        elseif startswith(strip(line), "#")
            write(io, replace(line, r"\s*#\s" => "", count = 1), '\n')
        else # code
            write(io, "```julia\n", line, "\n```\n")
        end
    end
    # collapse all adjacent code blocks
    replace(String(take!(io)), r"\n```\n+```julia" => "")
end
