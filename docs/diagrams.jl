# Generates the illustrations used in the topic guides as light/dark SVG pairs;
# docs/src/assets/custom.css shows the variant matching the active Documenter theme. One
# geometry definition per figure and two color themes, so the variants cannot drift apart.
#
# The output is committed, so this only needs running when a figure changes:
#   julia docs/diagrams.jl docs/src/topics/assets
#
# Colors come from the Okabe-Ito qualitative palette. Blue marks nodes and vertices, orange
# the things numbered per cell, and text is generic-family only, since an SVG loaded through
# <img> cannot use the page's webfonts.

struct Theme
    suffix::String
    ink::String        # primary lines and text
    muted::String      # secondary text, ticks, hatching
    node::String       # nodes / vertices
    cell::String       # cells / edges
    exact::String      # "reference"/exact quantity
    onaccent::String   # text on top of an accent-colored pill
    tint::String       # subtle area fill (accent at low opacity)
    tintop::Float64    # opacity of that fill
end

const LIGHT = Theme("-light", "#1c2024", "#5d6773", "#0072b2", "#d55e00", "#009e73", "#ffffff", "#0072b2", 0.08)
const DARK = Theme("-dark", "#dfe4ea", "#98a2b0", "#56b4e9", "#e69f00", "#00c9a0", "#12171c", "#56b4e9", 0.05)

const SANS = "Lato,'Helvetica Neue',Helvetica,Arial,sans-serif"
const MATH = "'Latin Modern Math','STIX Two Math','DejaVu Serif',Georgia,serif"

# --- primitives ---------------------------------------------------------------

f(x) = string(round(x; digits = 2))

function header(io, w, h, t)
    print(
        io, """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 $w $h" width="$w" height="$h">
        <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
                orient="auto-start-reverse"><path d="M0,0.6 L10,5 L0,9.4 z" fill="$(t.ink)"/></marker>
        <pattern id="hatch" width="7" height="7" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
        <line x1="0" y1="0" x2="0" y2="7" stroke="$(t.muted)" stroke-width="1.1"/></pattern>
        </defs>
        """
    )
    return
end

line(io, x1, y1, x2, y2; c, w = 2.0, dash = "", arrow = false) = print(
    io, """<line x1="$(f(x1))" y1="$(f(y1))" x2="$(f(x2))" y2="$(f(y2))" stroke="$c" """,
    """stroke-width="$w" stroke-linecap="round"$(dash == "" ? "" : " stroke-dasharray=\"$dash\"")""",
    arrow ? """ marker-end="url(#arrow)"/>\n""" : "/>\n"
)

dot(io, x, y, r; c) = print(io, """<circle cx="$(f(x))" cy="$(f(y))" r="$r" fill="$c"/>\n""")

path(io, d; c, w = 2.0, fill = "none", dash = "") = print(
    io, """<path d="$d" fill="$fill" stroke="$c" stroke-width="$w" stroke-linejoin="round" """,
    """stroke-linecap="round"$(dash == "" ? "" : " stroke-dasharray=\"$dash\"")/>\n"""
)

rect(io, x, y, w, h; fill, stroke = "none", sw = 2.0, rx = 0, op = 1) = print(
    io, """<rect x="$(f(x))" y="$(f(y))" width="$(f(w))" height="$(f(h))" rx="$rx" fill="$fill" """,
    """fill-opacity="$op" stroke="$stroke" stroke-width="$sw"/>\n"""
)

# `y` is the text baseline; `size` the font size in user units.
function text(io, x, y, s; c, size = 16, anchor = "middle", family = SANS, weight = "400", style = "normal")
    print(
        io, """<text x="$(f(x))" y="$(f(y))" fill="$c" font-family="$family" font-size="$size" """,
        """font-weight="$weight" font-style="$style" text-anchor="$anchor">$s</text>\n"""
    )
    return
end

sub(base, s) = "$base<tspan font-size=\"0.7em\" dy=\"0.22em\">$s</tspan>"

# Hatted symbol with a subscript, e.g. x̂₁ (U+0302 combining circumflex).
hat(base, s) = sub(base * "̂", s)

# Number in an accent-colored pill, centered on (x, y).
function pill(io, x, y, s; fill, fg, size = 15, w = 26, h = 21)
    rect(io, x - w / 2, y - h / 2, w, h; fill, rx = h / 2)
    text(io, x, y + size * 0.35, s; c = fg, size, weight = "700")
    return
end

function legend_entry(io, x, y, label, t; kind, size = 14)
    if kind === :node
        dot(io, x + 7, y, 6.0; c = t.node)
    elseif kind === :cell
        pill(io, x + 11, y, "n"; fill = t.cell, fg = t.onaccent, size = 12, w = 22, h = 18)
    end
    text(io, x + 28, y + size * 0.36, label; c = t.muted, size, anchor = "start")
    return
end

# --- figures ------------------------------------------------------------------

"""
Reference quadrilateral: vertex and edge numbering on [-1, 1]².
"""
function reference_quadrilateral(io, t)
    W, H = 400, 372
    header(io, W, H, t)
    x0, x1, y0, y1 = 82.0, 318.0, 62.0, 298.0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    rect(io, x0, y0, x1 - x0, y1 - y0; fill = t.tint, op = t.tintop)
    path(io, "M$(f(x0)),$(f(y0)) H$(f(x1)) V$(f(y1)) H$(f(x0)) Z"; c = t.ink, w = 2.2)

    # local coordinate system at the center
    line(io, cx, cy, cx + 74, cy; c = t.ink, w = 1.6, arrow = true)
    line(io, cx, cy, cx, cy - 74; c = t.ink, w = 1.6, arrow = true)
    text(io, cx + 84, cy + 6, sub("ξ", "1"); c = t.ink, size = 19, anchor = "start", family = MATH, style = "italic")
    text(io, cx + 7, cy - 80, sub("ξ", "2"); c = t.ink, size = 19, anchor = "start", family = MATH, style = "italic")

    # vertices, numbered diagonally outwards
    for (i, (vx, vy, dx, dy, an)) in enumerate(
            [
                (x0, y1, -17, 24, "end"), (x1, y1, 17, 24, "start"),
                (x1, y0, 17, -14, "start"), (x0, y0, -17, -14, "end"),
            ]
        )
        dot(io, vx, vy, 7.0; c = t.node)
        text(io, vx + dx, vy + dy, string(i); c = t.node, size = 21, anchor = an, weight = "700")
    end

    # edges, numbered just outside the edge midpoint
    for (i, (ex, ey)) in enumerate([(cx, y1 + 22), (x1 + 22, cy), (cx, y0 - 22), (x0 - 22, cy)])
        pill(io, ex, ey, string(i); fill = t.cell, fg = t.onaccent)
    end

    legend_entry(io, 92, H - 22, "vertex n", t; kind = :node)
    legend_entry(io, 212, H - 22, "edge n", t; kind = :cell)
    return print(io, "</svg>\n")
end

"""
A 2 x 2 quadrilateral grid with global node and cell numbering.
"""
function global_mesh(io, t)
    W, H = 430, 378
    header(io, W, H, t)
    xs = (108.0, 226.0, 344.0)
    ys = (256.0, 152.0, 48.0)   # bottom to top

    for j in 1:2, i in 1:2
        rect(io, xs[i], ys[j + 1], xs[i + 1] - xs[i], ys[j] - ys[j + 1]; fill = t.tint, op = t.tintop)
    end
    for x in xs
        line(io, x, ys[1], x, ys[3]; c = t.ink, w = 2.0)
    end
    for y in ys
        line(io, xs[1], y, xs[3], y; c = t.ink, w = 2.0)
    end

    # node numbers, placed outside the grid where possible
    n = 0
    for (j, y) in enumerate(ys), (i, x) in enumerate(xs)
        n += 1
        dot(io, x, y, 7.0; c = t.node)
        lx, an = i == 1 ? (x - 13, "end") : (x + 13, "start")
        text(io, lx, j == 1 ? y + 25 : y - 12, string(n); c = t.node, size = 19, anchor = an, weight = "700")
    end

    c = 0
    for j in 1:2, i in 1:2
        c += 1
        pill(io, (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2, string(c); fill = t.cell, fg = t.onaccent, w = 28)
    end

    # global coordinate system
    ox, oy = 46.0, 330.0
    line(io, ox, oy, ox + 52, oy; c = t.ink, w = 1.6, arrow = true)
    line(io, ox, oy, ox, oy - 52; c = t.ink, w = 1.6, arrow = true)
    text(io, ox + 61, oy + 6, "x"; c = t.ink, size = 18, anchor = "start", family = MATH, style = "italic")
    text(io, ox - 3, oy - 59, "y"; c = t.ink, size = 18, anchor = "middle", family = MATH, style = "italic")

    legend_entry(io, 122, H - 16, "node n", t; kind = :node)
    legend_entry(io, 242, H - 16, "cell n", t; kind = :cell)
    return print(io, "</svg>\n")
end

"""
Geometric mapping from the reference cell to the physical cell.
"""
function fe_mapping(io, t)
    W, H = 620, 274
    header(io, W, H, t)

    # reference cell
    rx0, rx1, ry0, ry1 = 78.0, 198.0, 66.0, 186.0
    rcx, rcy = (rx0 + rx1) / 2, (ry0 + ry1) / 2
    rect(io, rx0, ry0, rx1 - rx0, ry1 - ry0; fill = t.tint, op = t.tintop)
    path(io, "M$(f(rx0)),$(f(ry0)) H$(f(rx1)) V$(f(ry1)) H$(f(rx0)) Z"; c = t.ink, w = 2.2)
    line(io, rcx, rcy, rx1 + 26, rcy; c = t.muted, w = 1.4, arrow = true)
    line(io, rcx, rcy, rcx, ry0 - 26; c = t.muted, w = 1.4, arrow = true)
    text(io, rx1 + 32, rcy + 5, sub("ξ", "1"); c = t.muted, size = 17, anchor = "start", family = MATH, style = "italic")
    text(io, rcx + 6, ry0 - 30, sub("ξ", "2"); c = t.muted, size = 17, anchor = "start", family = MATH, style = "italic")
    for (i, (vx, vy, dx, dy, an)) in enumerate(
            [
                (rx0, ry1, -9, 22, "end"), (rx1, ry1, 9, 22, "start"),
                (rx1, ry0, 11, -10, "start"), (rx0, ry0, -11, -10, "end"),
            ]
        )
        dot(io, vx, vy, 6.0; c = t.node)
        text(
            io, vx + dx, vy + dy, hat("ξ", i); c = t.node, size = 18, anchor = an,
            family = MATH, style = "italic", weight = "700"
        )
    end
    text(io, rcx, H - 14, "reference cell"; c = t.muted, size = 15)

    # physical cell (curved edges from a bilinear/quadratic geometry)
    P = [(392.0, 208.0), (528.0, 182.0), (556.0, 68.0), (416.0, 52.0)]
    d = "M$(f(P[1][1])),$(f(P[1][2])) " *
        "Q460,190 $(f(P[2][1])),$(f(P[2][2])) " *
        "Q556,132 $(f(P[3][1])),$(f(P[3][2])) " *
        "Q482,84 $(f(P[4][1])),$(f(P[4][2])) " *
        "Q386,132 $(f(P[1][1])),$(f(P[1][2])) Z"
    print(io, """<path d="$d" fill="$(t.tint)" fill-opacity="$(t.tintop)" stroke="none"/>\n""")
    path(io, d; c = t.ink, w = 2.4)
    for (i, ((vx, vy), (dx, dy, an))) in enumerate(
            zip(P, [(0, 24, "middle"), (14, 16, "start"), (14, -8, "start"), (-4, -14, "end")])
        )
        dot(io, vx, vy, 6.0; c = t.node)
        text(
            io, vx + dx, vy + dy, hat("x", i); c = t.node, size = 18, anchor = an,
            family = MATH, style = "italic", weight = "700"
        )
    end
    ox, oy = 336.0, 250.0
    line(io, ox, oy, ox + 46, oy; c = t.muted, w = 1.4, arrow = true)
    line(io, ox, oy, ox, oy - 46; c = t.muted, w = 1.4, arrow = true)
    text(io, ox + 52, oy + 5, sub("x", "1"); c = t.muted, size = 17, anchor = "start", family = MATH, style = "italic")
    text(io, ox - 7, oy - 50, sub("x", "2"); c = t.muted, size = 17, anchor = "end", family = MATH, style = "italic")
    text(io, 490, H - 14, "physical cell"; c = t.muted, size = 15)

    # mapping arrow
    path(io, "M232,104 Q290,66 348,104"; c = t.exact, w = 2.2)
    print(
        io, """<path d="M348,104 l-11.5,-4.6 l1.6,9.6 z" fill="$(t.exact)" transform="rotate(28 348 104)"/>\n"""
    )
    text(io, 290, 56, "x(ξ)"; c = t.exact, size = 19, family = MATH, style = "italic", weight = "700")
    return print(io, "</svg>\n")
end

"""
`Dirichlet` vs `ProjectedDirichlet` for a prescribed function that the
interpolation cannot represent exactly.
"""
function projected_dirichlet(io, t)
    W, H = 620, 300
    header(io, W, H, t)
    xa, xb = 76.0, 336.0
    base, bot = 218.0, 268.0

    # body with the prescribed facet as its top edge
    rect(io, xa, base, xb - xa, bot - base; fill = "url(#hatch)")
    rect(io, xa, base, xb - xa, bot - base; fill = "none", stroke = t.muted, sw = 1.4)
    line(io, xa, base, xb, base; c = t.ink, w = 3.0)

    # f axis
    line(io, xa, bot, xa, 48; c = t.ink, w = 2.0, arrow = true)
    text(io, xa + 12, 62, "f(x)"; c = t.ink, size = 20, anchor = "start", family = MATH, style = "italic")

    # prescribed function (quadratic), its nodal interpolant, and its L2 projection
    path(io, "M$(f(xa)),125 Q206,72.5 $(f(xb)),200"; c = t.exact, w = 3.0)
    line(io, xa, 125, xb, 200; c = t.cell, w = 3.0, dash = "1 7")
    line(io, xa, 95, xb, 170; c = t.node, w = 3.0, dash = "1 7")
    for (y1, y2, c) in [(125, 200, t.cell), (95, 170, t.node)]
        dot(io, xa, y1, 7.0; c)
        dot(io, xb, y2, 7.0; c)
    end
    dot(io, xa, base, 6.0; c = t.ink)
    dot(io, xb, base, 6.0; c = t.ink)

    # legend
    lx = 386.0
    for (i, (label, c, dash)) in enumerate(
            [
                ("prescribed function", t.exact, ""),
                ("ProjectedDirichlet", t.node, "1 7"),
                ("Dirichlet", t.cell, "1 7"),
            ]
        )
        y = 108.0 + 38 * (i - 1)
        line(io, lx, y, lx + 46, y; c, w = 3.0, dash)
        text(io, lx + 58, y + 5, label; c = t.ink, size = 16, anchor = "start")
    end
    return print(io, "</svg>\n")
end

# --- topology figures ---------------------------------------------------------

# Node positions of an n x n quadrilateral grid; index 1 is the bottom/left.
topo_xs(x0, s, n) = [x0 + (i - 1) * s for i in 1:(n + 1)]
topo_ys(ybot, s, n) = [ybot - (j - 1) * s for j in 1:(n + 1)]

function topo_gridlines(io, xs, ys, t; c = t.ink, w = 2.0)
    for x in xs
        line(io, x, ys[1], x, ys[end]; c, w)
    end
    for y in ys
        line(io, xs[1], y, xs[end], y; c, w)
    end
    return
end

# Center of cell number c (cells numbered row-major from the bottom left).
function topo_cellcenter(xs, ys, c)
    n = length(xs) - 1
    i, j = (c - 1) % n + 1, (c - 1) ÷ n + 1
    return (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
end

"""
Exclusive neighborhood classification: the neighbors of cell 5 in a 3 x 3 grid,
colored by the highest-dimensional entity they share with cell 5.
"""
function topology_cell_neighbors(io, t)
    W, H = 430, 424
    header(io, W, H, t)
    xs = topo_xs(65.0, 100.0, 3)
    ys = topo_ys(350.0, 100.0, 3)

    for c in (2, 4, 6, 8) # sharing an edge with cell 5
        cx, cy = topo_cellcenter(xs, ys, c)
        rect(io, cx - 50, cy - 50, 100, 100; fill = t.exact, op = 0.18)
    end
    for c in (1, 3, 7, 9) # sharing only a vertex with cell 5
        cx, cy = topo_cellcenter(xs, ys, c)
        rect(io, cx - 50, cy - 50, 100, 100; fill = t.node, op = 0.18)
    end
    rect(io, xs[2], ys[3], 100, 100; fill = "url(#hatch)")
    topo_gridlines(io, xs, ys, t)
    for c in 1:9
        cx, cy = topo_cellcenter(xs, ys, c)
        pill(io, cx, cy, string(c); fill = t.cell, fg = t.onaccent, w = 28)
    end

    y = H - 20
    rect(io, 40, y - 7, 14, 14; fill = "url(#hatch)", stroke = t.ink, sw = 1.2)
    text(io, 62, y + 5, "the cell"; c = t.muted, size = 14, anchor = "start")
    rect(io, 140, y - 7, 14, 14; fill = t.exact, op = 0.3, stroke = t.exact, sw = 1.2)
    text(io, 162, y + 5, "edge neighbors"; c = t.muted, size = 14, anchor = "start")
    rect(io, 272, y - 7, 14, 14; fill = t.node, op = 0.3, stroke = t.node, sw = 1.2)
    text(io, 294, y + 5, "vertex neighbors"; c = t.muted, size = 14, anchor = "start")
    return print(io, "</svg>\n")
end

"""
One entity, several local views: an edge shared by two cells and a vertex shared
by four cells, addressed as (cell, local entity) index pairs.
"""
function topology_entity_views(io, t)
    W, H = 620, 306
    header(io, W, H, t)

    # left: the edge shared by cells 5 and 6
    x0, y0, s = 60.0, 95.0, 110.0
    rect(io, x0, y0, 2s, s; fill = t.tint, op = t.tintop)
    path(io, "M$(f(x0)),$(f(y0)) H$(f(x0 + 2s)) V$(f(y0 + s)) H$(f(x0)) Z"; c = t.ink, w = 2.2)
    line(io, x0 + s, y0, x0 + s, y0 + s; c = t.exact, w = 3.5)
    # local edge numbers, the two views of the shared edge highlighted
    for (k, hl) in ((0, 2), (1, 4)) # cell offset, highlighted local edge
        cx = x0 + k * s + s / 2
        for (i, (ex, ey)) in enumerate(
                [(cx, y0 + s - 8), (x0 + (k + 1) * s - 12, y0 + s / 2 + 4), (cx, y0 + 18), (x0 + k * s + 12, y0 + s / 2 + 4)]
            )
            c, weight = i == hl ? (t.exact, "700") : (t.muted, "400")
            text(io, ex, ey, string(i); c, size = 13, weight)
        end
    end
    pill(io, x0 + s / 2, y0 + s / 2, "5"; fill = t.cell, fg = t.onaccent)
    pill(io, x0 + 3s / 2, y0 + s / 2, "6"; fill = t.cell, fg = t.onaccent)
    text(io, x0 + s, 262, "EdgeIndex(5, 2) ↔ EdgeIndex(6, 4)"; c = t.ink, size = 15)
    text(io, x0 + s, 284, "two views of the same edge"; c = t.muted, size = 14)

    # right: the vertex shared by cells 5, 6, 8, and 9
    vx0, vy0, vs = 380.0, 60.0, 90.0
    vcx, vcy = vx0 + vs, vy0 + vs
    rect(io, vx0, vy0, 2vs, 2vs; fill = t.tint, op = t.tintop)
    path(io, "M$(f(vx0)),$(f(vy0)) H$(f(vx0 + 2vs)) V$(f(vy0 + 2vs)) H$(f(vx0)) Z"; c = t.ink, w = 2.2)
    line(io, vcx, vy0, vcx, vy0 + 2vs; c = t.ink, w = 2.0)
    line(io, vx0, vcy, vx0 + 2vs, vcy; c = t.ink, w = 2.0)
    # local number of the shared vertex in each of the four cells
    for (nr, dx, dy, an) in [(3, -12, 20, "end"), (4, 12, 20, "start"), (2, -12, -12, "end"), (1, 12, -12, "start")]
        text(io, vcx + dx, vcy + dy, string(nr); c = t.exact, size = 13, weight = "700", anchor = an)
    end
    dot(io, vcx, vcy, 7.5; c = t.exact)
    for (c, dx, dy) in [(5, -1, 1), (6, 1, 1), (8, -1, -1), (9, 1, -1)]
        pill(io, vcx + dx * vs / 2, vcy + dy * vs / 2, string(c); fill = t.cell, fg = t.onaccent)
    end
    text(io, vcx, 262, "VertexIndex: (5, 3), (6, 4), (8, 2), (9, 1)"; c = t.ink, size = 15)
    text(io, vcx, 284, "four views of the same vertex"; c = t.muted, size = 14)
    return print(io, "</svg>\n")
end

"""
The facet skeleton: each of the 24 unique facets of the 3 x 3 grid exactly once,
split into interior and boundary facets.
"""
function topology_facet_skeleton(io, t)
    W, H = 430, 410
    header(io, W, H, t)
    xs = topo_xs(65.0, 100.0, 3)
    ys = topo_ys(350.0, 100.0, 3)
    topo_gridlines(io, xs, ys, t; c = t.muted, w = 1.2)
    g = 9.0 # gap at the nodes so that each facet reads as its own segment
    for i in 1:4, j in 1:3 # vertical facets
        c = i in (2, 3) ? t.cell : t.exact
        line(io, xs[i], ys[j] - g, xs[i], ys[j + 1] + g; c, w = 3.5)
    end
    for j in 1:4, i in 1:3 # horizontal facets
        c = j in (2, 3) ? t.cell : t.exact
        line(io, xs[i] + g, ys[j], xs[i + 1] - g, ys[j]; c, w = 3.5)
    end
    for x in xs, y in ys
        dot(io, x, y, 3.0; c = t.muted)
    end

    y = H - 20
    line(io, 60, y, 94, y; c = t.cell, w = 3.5)
    text(io, 104, y + 5, "interior facet"; c = t.muted, size = 14, anchor = "start")
    line(io, 235, y, 269, y; c = t.exact, w = 3.5)
    text(io, 279, y + 5, "boundary facet"; c = t.muted, size = 14, anchor = "start")
    return print(io, "</svg>\n")
end

"""
A grid with mixed reference dimensions: a `Line` cell lying on the edge of a
`Quadrilateral`, drawn offset from the shared nodes for clarity.
"""
function topology_mixed_dim(io, t)
    W, H = 430, 400
    header(io, W, H, t)
    x0, x1, lx = 120.0, 260.0, 284.0
    ys = (50.0, 190.0, 330.0)

    rect(io, x0, ys[1], x1 - x0, ys[3] - ys[1]; fill = t.tint, op = t.tintop)
    path(io, "M$(f(x0)),$(f(ys[1])) H$(f(x1)) V$(f(ys[3])) H$(f(x0)) Z"; c = t.ink, w = 2.2)
    line(io, x0, ys[2], x1, ys[2]; c = t.ink, w = 2.0)

    # the line cell, offset from the edge it sits on
    line(io, lx, ys[1], lx, ys[2]; c = t.cell, w = 4.5)
    for y in (ys[1], ys[2])
        line(io, x1, y, lx, y; c = t.muted, w = 1.2, dash = "3 4")
        dot(io, lx, y, 5.5; c = t.node)
    end

    # global node numbers
    for (n, x, y, dx, dy, an) in [
            (1, x0, ys[3], -13, 6, "end"), (2, x1, ys[3], 13, 6, "start"),
            (3, x0, ys[2], -13, 6, "end"), (4, x1, ys[2], -8, 24, "end"),
            (5, x0, ys[1], -13, 6, "end"), (6, x1, ys[1], -8, -12, "end"),
        ]
        dot(io, x, y, 6.0; c = t.node)
        text(io, x + dx, y + dy, string(n); c = t.node, size = 18, anchor = an, weight = "700")
    end

    pill(io, (x0 + x1) / 2, (ys[1] + ys[2]) / 2, "1"; fill = t.cell, fg = t.onaccent, w = 28)
    pill(io, (x0 + x1) / 2, (ys[2] + ys[3]) / 2, "3"; fill = t.cell, fg = t.onaccent, w = 28)
    pill(io, lx, (ys[1] + ys[2]) / 2, "2"; fill = t.cell, fg = t.onaccent, w = 28)

    text(io, W / 2, 365, "cell 2 is a Line lying on the right edge of cell 1"; c = t.muted, size = 14)
    text(io, W / 2, 385, "(drawn offset for clarity)"; c = t.muted, size = 14)
    return print(io, "</svg>\n")
end

"""
The star (stencil) of a vertex: the vertex itself and all vertices connected to
it by an edge.
"""
function topology_vertex_star(io, t)
    W, H = 430, 410
    header(io, W, H, t)
    xs = topo_xs(65.0, 100.0, 3)
    ys = topo_ys(350.0, 100.0, 3)
    topo_gridlines(io, xs, ys, t; c = t.muted, w = 1.2)
    cx, cy = xs[3], ys[3]
    for (nx, ny) in ((xs[2], ys[3]), (xs[4], ys[3]), (xs[3], ys[2]), (xs[3], ys[4]))
        line(io, cx, cy, nx, ny; c = t.cell, w = 3.5)
        dot(io, nx, ny, 6.5; c = t.node)
    end
    dot(io, cx, cy, 8.0; c = t.exact)

    y = H - 20
    dot(io, 67, y, 7.0; c = t.exact)
    text(io, 82, y + 5, "center vertex"; c = t.muted, size = 14, anchor = "start")
    dot(io, 227, y, 6.5; c = t.node)
    text(io, 242, y + 5, "edge-connected vertices"; c = t.muted, size = 14, anchor = "start")
    return print(io, "</svg>\n")
end

# --- driver -------------------------------------------------------------------

const FIGURES = [
    "reference_quadrilateral" => reference_quadrilateral,
    "global_mesh" => global_mesh,
    "fe_mapping" => fe_mapping,
    "projected_dirichlet" => projected_dirichlet,
    "topology_cell_neighbors" => topology_cell_neighbors,
    "topology_entity_views" => topology_entity_views,
    "topology_facet_skeleton" => topology_facet_skeleton,
    "topology_mixed_dim" => topology_mixed_dim,
    "topology_vertex_star" => topology_vertex_star,
]

function main(dir = @__DIR__)
    for (name, fig) in FIGURES, t in (LIGHT, DARK)
        open(joinpath(dir, name * t.suffix * ".svg"), "w") do io
            fig(io, t)
        end
    end
    return
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main(isempty(ARGS) ? (@__DIR__) : ARGS[1])
end
