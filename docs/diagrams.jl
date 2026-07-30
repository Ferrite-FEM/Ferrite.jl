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

# --- driver -------------------------------------------------------------------

const FIGURES = [
    "reference_quadrilateral" => reference_quadrilateral,
    "global_mesh" => global_mesh,
    "fe_mapping" => fe_mapping,
    "projected_dirichlet" => projected_dirichlet,
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
