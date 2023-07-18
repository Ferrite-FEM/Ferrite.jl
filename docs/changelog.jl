const changelogfile = joinpath(@__DIR__, "../CHANGELOG.md")

function create_documenter_changelog()
    content = read(changelogfile, String)
    # Replace release headers
    content = replace(content, "## [Unreleased]" => "## Changes yet to be released")
    content = replace(content, r"## \[(\d+\.\d+\.\d+)\]" => s"## Version \1")
    # Replace [#XXX][github-XXX] with the proper links
    content = replace(content, r"(\[#(\d+)\])\[github-\d+\]" => s"\1(https://github.com/Ferrite-FEM/Ferrite.jl/pull/\2)")
    # Remove all links at the bottom
    content = replace(content, r"^<!-- Release links -->.*$"ms => "")
    # Change some GitHub in-readme links to documenter links
    content = replace(content, "(#upgrading-code-from-ferrite-03-to-10)" => "(@ref)")
    # Add a contents block
    last_sentence_before_content = "adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)."
    contents_block = """
    ```@contents
    Pages = ["changelog.md"]
    Depth = 2:2
    ```
    """
    content = replace(content, last_sentence_before_content => last_sentence_before_content * "\n\n" * contents_block)
    # Remove trailing lines
    content = strip(content) * "\n"
    # Write out the content
    write(joinpath(@__DIR__, "src/changelog.md"), content)
    return nothing
end

function fix_links()
    content = read(changelogfile, String)
    text = split(content, "<!-- Release links -->")[1]
    # Look for links of the form: [#XXX][github-XXX]
    github_links = Dict{String, String}()
    r = r"\[#(\d+)\](\[github-(\d+)\])"
    for m in eachmatch(r, text)
        @assert m[1] == m[3]
        # Always use /pull/ since it will redirect to /issues/ if it is an issue
        url = "https://github.com/Ferrite-FEM/Ferrite.jl/pull/$(m[1])"
        github_links[m[2]] = url
    end
    io = IOBuffer()
    print(io, "<!-- GitHub pull request/issue links -->\n\n")
    for l in sort!(collect(github_links); by = first)
        println(io, l[1], ": ", l[2])
    end
    content = replace(content, r"<!-- GitHub pull request/issue links -->.*$"ms => String(take!(io)))
    write(changelogfile, content)
end

if abspath(PROGRAM_FILE) == @__FILE__
    fix_links()
end
