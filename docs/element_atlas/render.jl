# Dependency-free SVG drawing. All fields passed to these routines are evaluated
# on the reference cell; projection is purely for display.
const PALETTES = (
    light = (bg = "#f7fafb", ink = "#243944", muted = "#607782", grid = "#d5e0e5", accent = "#007f86", warm = "#cf6943"),
    dark = (bg = "#17252e", ink = "#e2edf1", muted = "#9bb2bf", grid = "#354b58", accent = "#54d2c5", warm = "#ffab80"),
)
num(x) = iszero(round(x; digits = 3)) ? "0" : string(round(x; digits = 3))
esc(x) = replace(string(x), '&' => "&amp;", '<' => "&lt;", '>' => "&gt;", '"' => "&quot;")
xy(p) = "$(num(p[1])),$(num(p[2]))"
function line(io, a, b, color; width = 1.5, dash = "")
    return println(io, """<path d="M $(xy(a)) L $(xy(b))" fill="none" stroke="$color" stroke-width="$width" stroke-dasharray="$dash"/>""")
end
function poly(io, ps, fill; stroke = "none", width = 1)
    return println(io, """<polygon points="$(join(xy.(ps), ' '))" fill="$fill" stroke="$stroke" stroke-width="$width" stroke-linejoin="round"/>""")
end
function label(io, p, text, color; size = 15, anchor = "start")
    return println(io, """<text x="$(num(p[1]))" y="$(num(p[2]))" fill="$color" font-size="$size" text-anchor="$anchor">$(esc(text))</text>""")
end
function dotmark(io, p, color; r = 5, fill = color)
    return println(io, """<circle cx="$(num(p[1]))" cy="$(num(p[2]))" r="$r" fill="$fill" stroke="$color" stroke-width="2"/>""")
end
function arrow(io, a, b, color; width = 2)
    d = b .- a
    l = hypot(d...)
    l < 0.5 && return
    u = d ./ l
    v = (-u[2], u[1])
    h = min(6.0, l * 0.4)
    # Overlap the head near its base, not its tip: a full-length thick shaft
    # protrudes through the narrowing triangle. Scale tiny arrows consistently.
    line(io, a, b .- 0.9h .* u, color; width = min(width, 0.75h))
    return poly(io, [b, b .- h .* u .+ 0.45h .* v, b .- h .* u .- 0.45h .* v], color)
end
function svg(f, path, title, description, theme; height = 440)
    c = PALETTES[theme]
    return open(path, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 $height" role="img" aria-labelledby="title desc"><title id="title">$(esc(title))</title><desc id="desc">$(esc(description))</desc><rect width="960" height="$height" rx="16" fill="$(c.bg)"/><g font-family="system-ui, sans-serif">""")
        f(io, c)
        println(io, "</g></svg>")
    end
end
isquad(e) = Ferrite.getrefshape(e.ip) == RefQuadrilateral
isvector(e) = e.kind in (:normal, :tangent)
function mesh(e, n = 24)
    out = Vector{Vec{2, Float64}}[]
    if isquad(e)
        for j in 0:(n - 1), i in 0:(n - 1)
            a = Vec((-1 + 2i / n, -1 + 2j / n)); b = a + Vec((2 / n, 0.0))
            d = a + Vec((0.0, 2 / n)); c = b + Vec((0.0, 2 / n))
            push!(out, [a, b, c], [a, c, d])
        end
    else
        for j in 0:(n - 1), i in 0:(n - 1 - j)
            a = Vec((i / n, j / n)); b = a + Vec((1 / n, 0.0)); c = a + Vec((0.0, 1 / n))
            push!(out, [a, b, c])
            i + j < n - 1 && push!(out, [b, b + Vec((0.0, 1 / n)), c])
        end
    end
    return out
end
unitpoint(e, p) = isquad(e) ? ((p[1] + 1) / 2, (p[2] + 1) / 2) : (p[1], p[2])
function ramp(z, lim)
    t = clamp(z / lim, -1, 1)
    a = (227, 237, 234)
    b = t < 0 ? (102, 93, 165) : (0, 139, 142)
    return "rgb($(join(round.(Int, a .+ abs(t) .* (b .- a)), ',')))"
end
function surface(io, e, f, c; offset = 0, width = 540, lim = 1.0, title = "Scalar basis · height = value", nodes = false, selected = 0)
    label(io, (offset + 26, 34), title, c.ink; size = 17)
    project(p, z) = let (x, y) = unitpoint(e, p)
        (offset + width * (0.49 + 0.4(x - y)), 335 - 75(x + y) - 100z / lim)
    end
    vs = vertices(e)
    poly(io, project.(vs, 0), c.grid)
    for tri in sort(mesh(e); by = t -> -sum(p[1] + p[2] for p in t))
        vals = f.(tri)
        color = ramp(sum(vals) / 3, lim)
        poly(io, project.(tri, vals), color; stroke = color, width = 0.4)
    end
    # Sparse mesh lines give shape without overwhelming the smooth colour field.
    for tri in mesh(e, 6)
        ps = project.(tri, f.(tri))
        for k in 1:3
            line(io, ps[k], ps[mod1(k + 1, 3)], c.ink; width = 0.45)
        end
    end
    for k in eachindex(vs)
        a, b = vs[k], vs[mod1(k + 1, length(vs))]
        line(io, project(a, 0), project(b, 0), c.muted; dash = "4 4", width = 1)
    end
    origin = isquad(e) ? Vec((-1.0, -1.0)) : Vec((0.0, 0.0))
    ox = isquad(e) ? Vec((1.0, -1.0)) : Vec((1.0, 0.0))
    oy = isquad(e) ? Vec((-1.0, 1.0)) : Vec((0.0, 1.0))
    label(io, project(ox, 0) .+ (8, 17), "x", c.muted)
    label(io, project(oy, 0) .+ (-14, 17), "y", c.muted)
    label(io, project(origin, 0) .+ (0, 22), isquad(e) ? "(−1,−1)" : "(0,0)", c.muted; size = 12, anchor = "middle")
    if nodes
        for (i, p) in enumerate(Ferrite.reference_coordinates(e.ip))
            dotmark(io, project(p, f(p)), i == selected ? c.warm : c.ink; r = i == selected ? 6 : 3)
        end
    end
    for k in 0:49
        z = -lim + 2lim * k / 49
        println(io, """<rect x="$(offset + 120 + 4k)" y="388" width="4.2" height="8" fill="$(ramp(z, lim))"/>""")
    end
    label(io, (offset + 113, 398), num(-lim), c.muted; size = 12, anchor = "end")
    label(io, (offset + 328, 398), num(lim), c.muted; size = 12)
    return nothing
end
function field(io, e, f, c; offset = 0, scale = nothing, title = "Vector basis · arrows = direction and magnitude")
    label(io, (offset + 26, 34), title, c.ink; size = 16)
    project(p) = let (x, y) = unitpoint(e, p)
        (offset + 88 + 310x, 350 - 280y)
    end
    poly(io, project.(vertices(e)), c.bg; stroke = c.grid, width = 2)
    samples = isquad(e) ? [Vec((-1 + i / 5, -1 + j / 5)) for j in 1:9 for i in 1:9] : [Vec((i / 10, j / 10)) for j in 1:8 for i in 1:(9 - j)]
    magn = isnothing(scale) ? maximum(norm(f(p)) for p in samples) : scale
    for p in samples
        v = f(p)
        arrow(io, project(p), project(p) .+ (24v[1] / magn, -24v[2] / magn), c.accent; width = 1.8)
    end
    label(io, (offset + 92, 378), "0", c.muted; size = 13)
    label(io, (offset + 405, 364), "x", c.muted)
    label(io, (offset + 73, 64), "y", c.muted)
    arrow(io, (offset + 100, 403), (offset + 124, 403), c.accent)
    return label(io, (offset + 136, 408), "$(num(magn)) field units", c.muted; size = 13)
end
# Drawing and HTML hit targets share coordinates in the SVG's 960 × 440 viewBox.
const DOF_MARKER_CACHE = Dict{DataType, Any}()
dof_markers(e) = get!(() -> make_dof_markers(e), DOF_MARKER_CACHE, typeof(e.ip))
function make_dof_markers(e)
    project(p) = let (x, y) = unitpoint(e, p)
        (630 + 245x, 286 - 200y)
    end
    if isvector(e)
        markers = []
        for edge in edges(e)
            a, b = vertices(e)[collect(edge)]
            d = b - a
            v = e.kind == :normal ? Vec((d[2], -d[1])) / norm(d) : d / norm(d)
            for t in (weighted_edges(e) ? (0.32, 0.68) : (0.5,))
                base = project((1 - t) * a + t * b)
                direction = (v[1], -v[2])
                tip = base .+ 25 .* direction
                x, y = min.(base, tip) .- (16, 24)
                right, bottom = max.(base, tip) .+ (28, 20)
                push!(markers, (point = base, tip = tip, box = (x, y, right - x, bottom - y)))
            end
        end
        return markers
    end
    return map(Ferrite.reference_coordinates(e.ip)) do p
        x, y = project(p)
        return (point = (x, y), tip = nothing, box = (x - 18, y - 28, 44, 48))
    end
end
function layout(io, e, c, selected)
    label(io, (585, 34), "Degrees of freedom", c.ink; size = 17)
    project(p) = let (x, y) = unitpoint(e, p)
        (630 + 245x, 286 - 200y)
    end
    vs = vertices(e)
    poly(io, project.(vs), c.bg; stroke = c.grid, width = 2)
    for (i, p) in enumerate(vs)
        label(io, project(p) .+ (-10, 22), "v$i", c.muted; size = 11)
    end
    for (i, marker) in enumerate(dof_markers(e))
        color = i == selected ? c.warm : c.accent
        if isvector(e)
            arrow(io, marker.point, marker.tip, color; width = i == selected ? 3.5 : 2)
            label(io, marker.tip .+ (8, -5), i, color; size = 15)
        else
            dotmark(io, marker.point, color; r = i == selected ? 7 : 5)
            label(io, marker.point .+ (11, -10), i, color)
        end
    end
    label(io, (585, 338), isvector(e) ? "Arrows mark moments, not sampling points." : "Dots mark point evaluations; orange is selected.", c.muted; size = 13)
    return label(io, (585, 361), "Ferrite local numbering · reference cell", c.muted; size = 13)
end
function traces(io, e, f, g, c; selected = 0)
    label(io, (26, 32), "Edge densities · signed area gives the functional", c.ink; size = 17)
    n = Ferrite.getnbasefunctions(e.ip)
    for edge in 1:3
        offset = (edge - 1) * 320
        # BDM displays both weights in separate rows.
        ids = n == 6 ? ((2edge - 1):(2edge)) : (edge:edge)
        for (row, i) in enumerate(ids)
            ys = [density(e, f, i, s) for s in range(0, 1; length = 61)]
            zs = [density(e, g, i, s) for s in range(0, 1; length = 61)]
            lim = max(1.0, maximum(abs, ys), maximum(abs, zs)) * 1.12
            ybase = 124 + (row - 1) * 180
            project(s, v) = (offset + 53 + 225s, ybase - 49v / lim)
            line(io, project(0, 0), project(1, 0), c.muted)
            line(io, project(0, -lim), project(0, lim), c.grid)
            poly(io, [project(0, 0); [project(s, v) for (s, v) in zip(range(0, 1; length = 61), zs)]; project(1, 0)], c.grid)
            for (vals, color, dash) in ((ys, c.warm, "5 4"), (zs, c.accent, ""))
                ps = [project(s, v) for (s, v) in zip(range(0, 1; length = 61), vals)]
                println(io, """<polyline points="$(join(xy.(ps), ' '))" fill="none" stroke="$color" stroke-width="2.4" stroke-dasharray="$dash"/>""")
            end
            label(io, (offset + 53, ybase - 61), "edge $edge · ℓ$i", i == selected ? c.warm : c.ink; size = 13)
            label(io, project(0, lim) .+ (-7, 4), num(lim), c.muted; size = 10, anchor = "end")
            label(io, project(0, -lim) .+ (-7, 0), num(-lim), c.muted; size = 10, anchor = "end")
            label(io, (offset + 53, ybase + 67), "s = 0", c.muted; size = 11)
            label(io, (offset + 278, ybase + 67), "1", c.muted; size = 11, anchor = "end")
            label(io, (offset + 53, ybase + 87), "integral = $(num(functional(e, g, i)))", c.accent; size = 12)
        end
    end
    return
end
