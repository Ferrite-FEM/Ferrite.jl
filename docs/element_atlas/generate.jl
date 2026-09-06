module ElementAtlas
using Ferrite, LinearAlgebra
include("catalog.jl")
include("functionals.jl")
include("render.jl")

function picture(stem, alt; class = "", prefix = "assets")
    return """<img class="$class" src="$prefix/$stem-light.svg" alt="$(esc(alt))" loading="lazy"/><img class="$class" src="$prefix/$stem-dark.svg" alt="$(esc(alt))" loading="lazy"/>"""
end
function dof_controls(e, selected)
    return sprint() do io
        println(io, """<div class="atlas-dof-controls" role="group" aria-label="Choose a degree of freedom">""")
        for (i, marker) in enumerate(dof_markers(e))
            x, y, w, h = marker.box
            description = esc("Activate dof $i · $(Ferrite.dof_functionals(e.ip)[i])")
            println(io, """<button type="button" class="atlas-dof-button" data-dof="$i" aria-label="$description" title="$description" aria-pressed="$(i == selected)" style="left:$(num(100x / 960))%;top:$(num(100y / 440))%;width:$(num(100w / 960))%;height:$(num(100h / 440))%"></button>""")
        end
        println(io, "</div>")
    end
end
rawhtml(io, html) = println(io, "\n```@raw html\n$html\n```\n")
math(io, tex) = println(io, "\n```math\n$tex\n```\n")
constructor(e) = string(e.ip)
moment_caption(e) = "Density includes the edge-length factor and, for weighted moments, the linear weight. Shaded area is signed."
function generate_figures(e, assets)
    n = getnbasefunctions(e.ip)
    for theme in keys(PALETTES), i in 1:n
        f = basis(e, i)
        svg(joinpath(assets, "$(e.slug)-basis-$i-$theme.svg"), "$(e.title): basis $i", "$(e.summary) Selected dof $i is orange; all numbers use Ferrite local ordering.", theme) do io, c
            isvector(e) ? field(io, e, f, c) : surface(io, e, f, c; nodes = true, selected = i)
            layout(io, e, c, i)
        end
        if isvector(e)
            svg(joinpath(assets, "$(e.slug)-trace-$i-$theme.svg"), "Moments for basis $i", "$(moment_caption(e)) The selected functional integrates to one; all others integrate to zero.", theme; height = n == 6 ? 430 : 235) do io, c
                traces(io, e, f, f, c; selected = i)
            end
        end
    end
    return
end

function write_element(io, e; prettyurls = true)
    prefix = prettyurls ? "../assets" : "assets"
    pagepicture(stem, alt) = picture(stem, alt; prefix)
    n = getnbasefunctions(e.ip)
    rawhtml(io, """<h3>$(esc(e.summary))</h3><p class="atlas-tag">$(esc(e.continuity))</p>""")
    println(io, e.detail, "\n")
    cell = isquad(e) ? raw"[-1,1]^2" : raw"\{(x,y):x,y\geq0,\ x+y\leq1\}"
    cellname = isquad(e) ? "Quadrilateral" : "Triangle"
    println(io, "| Reference cell | Local dofs | Value shape | Order parameter |\n|:--|:--|:--|:--|\n| $cellname | $n | $(isvector(e) ? "2-vector" : "Scalar") | $(Ferrite.getorder(e.ip)) |\n")
    rawhtml(io, "<h4>Explore the basis</h4><p>Click a dof on the right or use the dropdown to see its dual basis function. The orange mark identifies the selected functional.</p>")
    rawhtml(io, """<div class="atlas-explorer"><div class="atlas-controls" hidden><label for="atlas-basis-$(e.slug)">Local basis</label><select id="atlas-basis-$(e.slug)" aria-label="Select local basis function">$(join(["<option value=\"$i\">Basis $i · $(esc(string(Ferrite.dof_functionals(e.ip)[i])))</option>" for i in 1:n]))</select><span class="atlas-counter" aria-live="polite"></span></div>""")
    for i in 1:n
        rawhtml(io, """<div class="atlas-basis" data-basis="$i"><figure class="atlas-figure"><div class="atlas-diagram">$(pagepicture("$(e.slug)-basis-$i", "$(e.title), basis $i: $(e.summary)"))$(i == 1 ? dof_controls(e, 1) : "")</div><figcaption>Basis $i · $(esc(string(Ferrite.dof_functionals(e.ip)[i])))</figcaption></figure>""")
        math(io, "\\ell_j(\\varphi_{$i})=\\delta_{j,$i},\\qquad j=1,\\ldots,$n.")
        if isvector(e)
            rawhtml(io, "<figure class=\"atlas-figure\">$(pagepicture("$(e.slug)-trace-$i", "Moments for basis $i; the selected moment is one, all others are zero."))<figcaption>$(moment_caption(e))</figcaption></figure>")
        end
        rawhtml(io, """<p class="atlas-download"><a href="$prefix/$(e.slug)-basis-$i-light.svg" download>Download SVG · light</a> / <a href="$prefix/$(e.slug)-basis-$i-dark.svg" download>dark</a></p></div>""")
    end
    rawhtml(io, "</div>")
    rawhtml(io, "<h4>Definition</h4><p>The reference cell is</p>")
    math(io, "\\widehat K = $cell.")
    println(io, "The local approximation space is")
    math(io, e.space)
    println(io, "The degrees of freedom are the linear functionals")
    math(io, e.functional)
    println(io, "The basis is dual to these functionals:")
    math(io, raw"\ell_i(\varphi_j)=\delta_{ij},\qquad I_h f=\sum_{i=1}^{n}\ell_i(f)\,\varphi_i.")
    if isvector(e)
        println(io, "Reference edges follow Ferrite's local vertex ordering. Linear edge weights use the parameter ``s\\in[0,1]`` from the first vertex to the second.")
        println(io, "\nUnder a cell map ``F`` with Jacobian ``J``, the basis uses the $(e.kind == :normal ? "contravariant" : "covariant") Piola transform:")
        math(io, e.kind == :normal ? raw"\boldsymbol v\circ F=\frac{1}{\det J}J\widehat{\boldsymbol v}" : raw"\boldsymbol v\circ F=J^{-\mathsf T}\widehat{\boldsymbol v}")
    end
    println(io, e.note)
    rawhtml(io, "<h4>Use in Ferrite</h4>")
    println(io, "```julia\nusing Ferrite\n\nip = $(constructor(e))\ngetnbasefunctions(ip)       # $n\nFerrite.dof_functionals(ip) # functional kinds in local dof order\n```\n")
    println(io, "See [dof functionals](@ref dof-functionals) for the API, [interpolations](@ref reference-interpolation) for supported variants, and [$(e.title) on DefElement](https://defelement.org/elements/$(e.defelement).html) for the wider family. Return to the [element atlas](@ref element-atlas).")
    return nothing
end

function page(e, out; prettyurls = true)
    return open(joinpath(out, e.slug * ".md"), "w") do io
        println(io, "# [$(e.title)](@id element-$(e.slug))\n")
        rawhtml(io, """<div class="atlas-kicker">ELEMENT ATLAS <span>Proof of concept</span></div>""")
        write_element(io, e; prettyurls)
    end
end

function generate(out = joinpath(@__DIR__, "..", "src", "elements"); prettyurls = true)
    check()
    assets = joinpath(out, "assets")
    mkpath(assets)
    for e in CATALOG
        generate_figures(e, assets)
        page(e, out; prettyurls)
    end
    open(joinpath(out, "index.md"), "w") do io
        println(io, "# [Element atlas](@id element-atlas)\n")
        rawhtml(io, """<div class="atlas-kicker">FERRITE / FINITE ELEMENTS <span>Proof of concept</span></div><p class="atlas-lead">A field is defined by what you measure.</p>""")
        println(io, "Explore the spaces, degrees of freedom, and basis functions behind Ferrite's interpolations. A point value, a normal flux, and a circulation integral call for different pictures.\n\nEach family has one representative example. The figures are SVGs generated from this checkout's basis functions, with light and dark variants.\n")
        rawhtml(io, """<div class="atlas-key"><span>● Point values → surfaces</span><span>↗ Moments → vector fields &amp; traces</span></div>""")
        for e in CATALOG
            thumb = e.slug == "bubble-enriched-lagrange" ? 4 : 1
            # Markdown links are resolved by Documenter for both pretty and flat URLs.
            rawhtml(io, "<div class=\"atlas-card\"><div class=\"atlas-card-image\">$(picture("$(e.slug)-basis-$thumb", "$(e.title) basis preview"))</div><div class=\"atlas-card-copy\"><span class=\"atlas-tag\">$(esc(e.continuity))</span>")
            println(io, "\n### [$(e.title)]($(e.slug).md)\n\n$(e.summary)\n\n`$(constructor(e))`\n")
            rawhtml(io, "</div></div>")
        end
        println(io, "## How to read the atlas\n\nA reference finite element combines a cell ``\\widehat K``, a finite-dimensional space ``V``, and a set of linear functionals ``\\ell_i``. The shape functions satisfy")
        math(io, raw"\ell_i(\varphi_j)=\delta_{ij},\qquad I_hf=\sum_i\ell_i(f)\varphi_i.")
        println(io, "Each page pairs a selectable basis with its dof support and mathematical definition. Moment elements use integral functionals rather than point samples. All examples are on the reference cell.\n\nThe numerical generator checks the complete duality matrix before writing any pages. Moment weights are explicitly defined in the generator: the functional-kind API alone does not specify them.\n\nThe presentation is inspired by [DefElement](https://defelement.org/). All figures here are generated locally from Ferrite. See the [interpolation API](@ref reference-interpolation) for the complete list of supported reference cells and orders.")
    end
    return ["Overview" => "elements/index.md"; [e.title => "elements/$(e.slug).md" for e in CATALOG]]
end
end # module
