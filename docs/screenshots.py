# Renders the example screenshots/animations with ParaView. Each scene is
# saved twice on a transparent background: <name>-light.png with dark
# annotations and <name>-dark.png with light ones; docs/src/assets/custom.css
# shows the right one for the active Documenter theme.
#
# The .vtu data files are produced by docs/generate_screenshots.jl. Usage:
#   julia --project=docs docs/generate_screenshots.jl [names...]
# which runs the examples and then invokes
#   pvbatch docs/screenshots.py <datadir> <outdir> [names...]
# With no names all scenes are rendered, otherwise only the selected ones.
import glob
import os
import re
import subprocess
import sys
import tempfile

from paraview.simple import *

try:
    # VTK typesets the "$...$" scalar bar titles through matplotlib. Without it they
    # render literally and the labels come out in a different size and weight, so bail
    # out rather than write assets that don't match the rest.
    import matplotlib  # noqa: F401
except ImportError:
    sys.exit(
        "matplotlib is not importable by pvbatch, which is needed to typeset the scalar\n"
        "bar titles. Install it into ParaView's Python, or install it elsewhere and point\n"
        "PYTHONPATH at that directory:\n"
        "    pip install --target=/tmp/pvlibs matplotlib\n"
        "    PYTHONPATH=/tmp/pvlibs julia --project=docs docs/generate_screenshots.jl ..."
    )

datadir, outdir = sys.argv[1], sys.argv[2]
selected = sys.argv[3:]
RES = [1600, 1200]
ANIM_RES = [1000, 750]  # animations are rendered directly at their final size
EDGE = [0.15, 0.15, 0.15]  # mesh edge color (reads on both light and dark surfaces)
BARS = []  # scalar bars whose text flips between the variants
ANNOTS = []  # displays drawn in the annotation color, which also flips
# (suffix, annotation color, background color): the two theme variants each
# scene is saved as. The saved images are transparent, but edge antialiasing
# blends against the render background, so match it to the docs theme to
# avoid light/dark halos around the mesh.
VARIANTS = (
    ("light", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    ("dark", [0.9, 0.9, 0.9], [0.122, 0.133, 0.161]),
)
ANIM_FRAMES = 40  # cap on frames per animation (timesteps are subsampled to fit)

SCENES = {}


def scene(name):
    def register(fn):
        SCENES[name] = fn
        return fn
    return register


def new_view():
    view = CreateRenderView()
    SetActiveView(view)
    view.ViewSize = RES
    view.UseColorPaletteForBackground = 0
    view.Background = [1.0, 1.0, 1.0]
    view.OrientationAxesVisibility = 0
    return view


def surface(source, view, edges=True):
    d = Show(source, view)
    if edges:
        d.Representation = "Surface With Edges"
        d.EdgeColor = EDGE
        d.LineWidth = 1.0
    else:
        d.Representation = "Surface"
    return d


def annotate(source, view, width=3.0):
    # Draw source as flat, unshaded geometry in the annotation color, e.g. glyphs
    # or a marked-up cut line. The color follows the theme variant, so unlike a
    # data-colored display it stays readable on both backgrounds.
    d = Show(source, view)
    d.Representation = "Surface"
    d.ColorArrayName = ["POINTS", ""]
    d.Ambient, d.Diffuse = 1.0, 0.0  # no headlight shading, so it reads as line art
    d.LineWidth = width
    ANNOTS.append(d)
    return d


def lift(source, dz=0.05):
    # Nudge geometry towards the camera of a 2d view, so overlays drawn on top of
    # a surface in the z = 0 plane don't z-fight with it.
    t = Transform(Input=source)
    t.Transform.Translate = [0.0, 0.0, dz]
    return t


def warp(source, array, factor):
    w = WarpByVector(Input=source, Vectors=["POINTS", array])
    w.ScaleFactor = factor
    return w


def open_series(pattern):
    # Open a numbered .vtu series (name_1.vtu, name_2.vtu, ...) as a time series.
    files = sorted(glob.glob(pattern),
                   key=lambda f: int(re.search(r"_(\d+)\.vtu$", f).group(1)))
    return OpenDataFile(files)


def colorbar(display, view, array, title=None, preset="Cool to Warm",
             horizontal=False, pos=None, fmt=None, labels=None, categories=0,
             length=None):
    # categories: the array holds a category index 1..n rather than a magnitude (e.g. a
    # coloring), so give each one its own swatch from a qualitative preset. The indices
    # themselves carry no order, so the swatches are drawn without tick labels.
    ColorBy(display, array)
    display.SetScalarBarVisibility(view, True)
    lut = GetColorTransferFunction(array[1])
    if categories:
        lut.ApplyPreset(preset if preset != "Cool to Warm" else "Brewer Qualitative Set1", True)
        lut.InterpretValuesAsCategories = 1
        lut.Annotations = [s for i in range(1, categories + 1) for s in (str(i), "")]
    else:
        lut.ApplyPreset(preset, True)
    bar = GetScalarBar(lut, view)
    bar.Title = title or array[1]
    bar.ComponentTitle = ""
    if horizontal:  # wide domains read better with the bar centred below the mesh
        bar.Orientation = "Horizontal"
        if pos is None:
            bar.WindowLocation = "Lower Center"
        else:
            bar.WindowLocation = "Any Location"
            bar.Position = pos
        if length is not None:
            bar.ScalarBarLength = length
    else:
        # right of the mesh, vertically centred, with the title on top
        bar.Orientation = "Vertical"
        bar.HorizontalTitle = 1
        bar.WindowLocation = "Any Location"
        bar.Position = pos or [0.83, 0.32]
        bar.ScalarBarLength = length or 0.36
    bar.ScalarBarThickness = 24
    bar.TitleFontSize = 36
    bar.LabelFontSize = 32
    bar.TitleBold = 1
    bar.LabelBold = 1
    if fmt is not None:
        # e.g. "%.0f": plain numbers for the min/max labels where the default
        # exponent format grows too wide (the in-between labels stay automatic)
        bar.RangeLabelFormat = fmt
    if categories:
        bar.DrawTickMarks = bar.DrawTickLabels = bar.DrawAnnotations = 0
    if labels is not None:
        # Fixed in-between labels, shown alongside the min/max ones. Use where the
        # automatic labels crowd together, which also makes the tick spacing
        # independent of how many labels the ParaView version decides to fit.
        bar.UseCustomLabels = 1
        bar.CustomLabels = labels
    BARS.append(bar)
    return lut


def _set_camera(view, azimuth, elevation, zoom, twod, pan_y=0.0, bounds=None,
                parallel=False):
    # bounds: frame these explicit bounds instead of the currently shown data
    # (see bounds_over_time), e.g. so an oscillating geometry stays in view.
    # parallel: orthographic projection for the 3d camera, e.g. so copies of a
    # domain stacked for comparison keep their alignment and relative size.
    reset = (lambda: view.ResetCamera(*bounds, False)) if bounds else (lambda: view.ResetCamera(False))
    reset()
    cam = GetActiveCamera()
    if twod:
        view.CameraParallelProjection = 1
    else:
        cam.Azimuth(azimuth)
        cam.Elevation(elevation)
        if parallel:
            view.CameraParallelProjection = 1
    reset()
    cam.Zoom(zoom)
    # pan_y < 0 shifts the scene up in the frame, freeing space for a
    # horizontal colour bar below the mesh
    cam.SetWindowCenter(0.0, pan_y)


def _apply_variant(view, text, bg):
    view.Background = bg
    for bar in BARS:
        bar.TitleColor = text
        bar.LabelColor = text
    for d in ANNOTS:
        d.AmbientColor = text
        d.DiffuseColor = text


def finish(view, name, azimuth=30, elevation=25, zoom=1.0, twod=False, res=None,
           pan_y=0.0, parallel=False):
    # res overrides the frame size (default RES); use a matching aspect ratio for
    # non-square domains so the scene fills the frame instead of leaving margins.
    res = res or RES
    view.ViewSize = res
    Render()  # settle the render window / scalar-bar layout at the new size first
    _set_camera(view, azimuth, elevation, zoom, twod, pan_y, parallel=parallel)
    for variant, text, bg in VARIANTS:
        _apply_variant(view, text, bg)
        Render()
        SaveScreenshot(
            outdir + "/" + name + "-" + variant + ".png", view,
            ImageResolution=res, TransparentBackground=1,
        )
    BARS.clear()
    ANNOTS.clear()
    Delete(view)


def cell_range(source, name):
    info = source.GetDataInformation().GetCellDataInformation().GetArrayInformation(name)
    return info.GetComponentRange(0)


def data_range_over_time(source, array, comp=None, times=None):
    # comp: component index (default: magnitude for vectors, else the scalar).
    # times: timestep list; required when source is a filter (only readers
    # carry TimestepValues).
    assoc, name = array
    lo, hi = float("inf"), float("-inf")
    for t in (times if times is not None else list(source.TimestepValues) or [0.0]):
        source.UpdatePipeline(t)
        di = source.GetDataInformation()
        info = (di.GetPointDataInformation() if assoc == "POINTS"
                else di.GetCellDataInformation()).GetArrayInformation(name)
        c = comp if comp is not None else (-1 if info.GetNumberOfComponents() > 1 else 0)
        rng = info.GetComponentRange(c)
        lo, hi = min(lo, rng[0]), max(hi, rng[1])
    return lo, hi


def bounds_over_time(source, times):
    # Union of the geometry bounds over the timesteps, e.g. of a warp filter
    # whose output moves around (readers carry TimestepValues, filters need
    # them passed in). Pass the result as frame_bounds to finish_anim when no
    # single timestep covers the whole motion.
    b = [float("inf"), float("-inf")] * 3
    for t in times:
        source.UpdatePipeline(t)
        tb = source.GetDataInformation().GetBounds()
        for i in range(3):
            b[2 * i] = min(b[2 * i], tb[2 * i])
            b[2 * i + 1] = max(b[2 * i + 1], tb[2 * i + 1])
    return b


def finish_anim(view, source, name, azimuth=30, elevation=25, zoom=1.0,
                twod=False, delay=8, res=None, pan_y=0.0, frame_bounds=None):
    # Frames are rendered directly at the final animation size (res, default
    # ANIM_RES; use a matching aspect ratio for non-square domains) --
    # downscaling afterwards would blur the annotation text. Saved as animated
    # WebP: full 24-bit colour and 8-bit alpha, where GIF's 256 colours and
    # 1-bit transparency butchered the gradients and text edges.
    res = res or ANIM_RES
    view.ViewSize = res
    times = list(source.TimestepValues) or [0.0]
    if len(times) > ANIM_FRAMES:
        step = -(-len(times) // ANIM_FRAMES)  # ceil, so the result stays <= ANIM_FRAMES
        times = times[::step]
    # Frame once at the last step (largest extent for deforming geometry),
    # unless explicit frame_bounds are given.
    view.ViewTime = times[-1]
    Render()  # settle the render window / scalar-bar layout at the new size first
    _set_camera(view, azimuth, elevation, zoom, twod, pan_y, bounds=frame_bounds)
    for variant, text, bg in VARIANTS:
        _apply_variant(view, text, bg)
        with tempfile.TemporaryDirectory() as frames:
            for i, t in enumerate(times):
                view.ViewTime = t
                Render()
                SaveScreenshot(
                    "%s/f%04d.png" % (frames, i), view,
                    ImageResolution=res, TransparentBackground=1,
                )
            out = outdir + "/" + name + "-" + variant + ".webp"
            print("  encoding %s (%d frames, slow)" % (os.path.basename(out), len(times)),
                  flush=True)
            # High-quality lossy: lossless blows up on gradient-heavy scenes
            # (>10MB); sharp-yuv keeps the annotation text edges crisp.
            subprocess.run(
                ["magick", "-delay", str(delay), "-loop", "0"]
                + sorted(glob.glob(frames + "/f*.png"))
                + ["-quality", "90", "-define", "webp:use-sharp-yuv=1",
                   "-define", "webp:alpha-quality=100", "-define", "webp:method=6",
                   out],
                check=True,
            )
    BARS.clear()
    ANNOTS.clear()
    Delete(view)


# --- heat_equation: temperature on the unit square
@scene("heat_equation")
def scene_heat_equation():
    view = new_view()
    r = OpenDataFile(datadir + "/heat_equation.vtu")
    d = surface(r, view)
    colorbar(d, view, ("POINTS", "u"), title="u")
    finish(view, "heat_equation", twod=True, zoom=0.95)


# --- plasticity: von Mises stress on a deformed cantilever beam
@scene("plasticity")
def scene_plasticity():
    view = new_view()
    r = OpenDataFile(datadir + "/plasticity.vtu")
    w = warp(r, "u", 25.0)
    d = surface(w, view)
    colorbar(d, view, ("CELLS", "von Mises [Pa]"), title="von Mises [Pa]",
             horizontal=True, pos=[0.335, 0.14])  # right under the beam
    # wide beam -> wide frame, cuts the empty space above/below
    finish(view, "plasticity", azimuth=-35, elevation=20, zoom=1.5, res=[1600, 800])


# --- hyperelasticity: twisted cube coloured by displacement magnitude
@scene("hyperelasticity")
def scene_hyperelasticity():
    view = new_view()
    r = OpenDataFile(datadir + "/hyperelasticity.vtu")
    w = warp(r, "u", 1.0)
    d = surface(w, view, edges=False)
    colorbar(d, view, ("POINTS", "u"), title="$\\vert u \\vert$")
    finish(view, "hyperelasticity", azimuth=35, elevation=20, zoom=1.1)


# --- dg_heat_equation: temperature on the unit square
@scene("dg_heat_equation")
def scene_dg_heat_equation():
    view = new_view()
    r = OpenDataFile(datadir + "/dg_heat_equation.vtu")
    d = surface(r, view, edges=False)
    colorbar(d, view, ("POINTS", "u"), title="T")
    finish(view, "dg_heat_equation", twod=True, zoom=0.95)


# --- linear_elasticity: deformed Ferrite logo coloured by vertical stress
@scene("linear_elasticity")
def scene_linear_elasticity():
    view = new_view()
    r = OpenDataFile(datadir + "/linear_elasticity.vtu")
    w = warp(r, "u", 2.0)
    d = surface(w, view)
    colorbar(d, view, ("CELLS", "sigma_22"), title="$\\sigma_{22}$")
    finish(view, "linear_elasticity", twod=True, zoom=0.95)

    # Figure 2 of the tutorial: the L2-projected nodal stress (left) next to
    # the constant average stress per cell (right), on one shared colour scale.
    view = new_view()
    r = OpenDataFile(datadir + "/linear_elasticity.vtu")
    left = surface(warp(r, "u", 2.0), view)
    ColorBy(left, ("POINTS", "stress field"))
    shifted = Transform(Input=warp(r, "u", 2.0))
    shifted.Transform.Translate = [1.1, 0.0, 0.0]
    right = surface(shifted, view)
    lut_cell = colorbar(right, view, ("CELLS", "sigma_22"),
                        title="$\\sigma_{22}$", horizontal=True,
                        labels=[5000.0, 10000.0, 15000.0])
    lut_proj = GetColorTransferFunction("stress field")
    lut_proj.ApplyPreset("Cool to Warm", True)
    lut_proj.VectorMode = "Component"
    lut_proj.VectorComponent = 1  # yy
    # one scale for both fields: rescale the two LUTs to the union range
    di = r.GetDataInformation()
    pr = di.GetPointDataInformation().GetArrayInformation("stress field").GetComponentRange(1)
    cr = di.GetCellDataInformation().GetArrayInformation("sigma_22").GetComponentRange(0)
    lo, hi = min(pr[0], cr[0]), max(pr[1], cr[1])
    lut_proj.RescaleTransferFunction(lo, hi)
    lut_cell.RescaleTransferFunction(lo, hi)
    finish(view, "linear_elasticity_stress", twod=True, zoom=1.3, res=[1600, 950])


# --- incompressible_elasticity: von Mises stress on the deformed Cook membrane
@scene("incompressible_elasticity")
def scene_incompressible_elasticity():
    view = new_view()
    r = OpenDataFile(datadir + "/cook_quadratic_linear.vtu")
    w = warp(r, "u", 1.0)
    d = surface(w, view, edges=False)
    colorbar(d, view, ("CELLS", "sigma von Mises"), title="von Mises")
    finish(view, "incompressible_elasticity", twod=True, zoom=0.95)


# --- stokes-flow: velocity magnitude on the quarter disk
@scene("stokes-flow")
def scene_stokes_flow():
    view = new_view()
    r = OpenDataFile(datadir + "/stokes-flow.vtu")
    d = surface(r, view, edges=False)
    colorbar(d, view, ("POINTS", "u"), title="speed")
    finish(view, "stokes-flow", twod=True, zoom=0.95)


# --- computational_homogenization: von Mises stress on the deformed RVE,
# comparing Dirichlet (left) and periodic (right) boundary conditions for the
# shear load case, on one shared colour scale (matching the Figure 1 caption)
@scene("computational_homogenization")
def scene_computational_homogenization():
    view = new_view()
    r = OpenDataFile(datadir + "/homogenization.vtu")
    left = surface(warp(r, "u_dirichlet_3", 0.3), view, edges=False)
    ColorBy(left, ("POINTS", "σvM_dirichlet_3"))
    lut_d = GetColorTransferFunction("σvM_dirichlet_3")
    lut_d.ApplyPreset("Cool to Warm", True)
    b = r.GetDataInformation().GetBounds()
    shifted = Transform(Input=warp(r, "u_periodic_3", 0.3))
    shifted.Transform.Translate = [(b[1] - b[0]) * 1.35, 0.0, 0.0]
    right = surface(shifted, view, edges=False)
    lut_p = colorbar(right, view, ("POINTS", "σvM_periodic_3"),
                     title="von Mises [Pa]", horizontal=True,
                     labels=[2.0e10, 4.0e10, 6.0e10])
    pd = r.GetDataInformation().GetPointDataInformation()
    hi = max(pd.GetArrayInformation("σvM_dirichlet_3").GetComponentRange(0)[1],
             pd.GetArrayInformation("σvM_periodic_3").GetComponentRange(0)[1])
    lut_d.RescaleTransferFunction(0.0, hi)
    lut_p.RescaleTransferFunction(0.0, hi)
    finish(view, "computational_homogenization", twod=True, zoom=1.25,
           res=[1600, 950], pan_y=-0.12)


# --- linear_shell: deflected shell coloured by displacement magnitude
@scene("linear_shell")
def scene_linear_shell():
    view = new_view()
    r = OpenDataFile(datadir + "/linear_shell.vtu")
    w = warp(r, "u", 2.0)
    d = surface(w, view)
    d.Ambient = 0.25  # lift the headlight-only shading, which reads too dark
    colorbar(d, view, ("POINTS", "u"), title="$\\vert u \\vert$")
    # moderate warp and a view from above so it reads as a flat plate (a
    # grazing view also makes the headlight shading unnaturally dark)
    finish(view, "linear_shell", azimuth=30, elevation=65, zoom=1.0)


# --- darcy_flow: pressure (left) next to flux magnitude (right), showing the
# flow forced through the gap in the almost impermeable barrier
@scene("darcy_flow")
def scene_darcy_flow():
    view = new_view()
    r = OpenDataFile(datadir + "/darcy_flow.vtu")
    left = surface(r, view, edges=False)
    colorbar(left, view, ("CELLS", "p"), title="p", pos=[0.02, 0.32], fmt="%.1f")
    shifted = Transform(Input=r)
    shifted.Transform.Translate = [1.1, 0.0, 0.0]
    right = surface(shifted, view, edges=False)
    colorbar(right, view, ("POINTS", "q"), title="$\\vert q \\vert$",
             pos=[0.87, 0.32], fmt="%.1f")
    # arrows on the flux panel showing the flow direction through the gap,
    # on a regular lattice (resampled) so the arrangement reads calmly
    lattice = ResampleToImage(Input=shifted)
    lattice.UseInputBounds = 0
    lattice.SamplingBounds = [1.135, 2.065, 0.035, 0.965, 0.0, 0.0]
    lattice.SamplingDimensions = [13, 13, 1]
    g = Glyph(Input=lattice, GlyphType="Arrow")
    g.OrientationArray = ["POINTS", "q"]
    g.ScaleArray = ["POINTS", "q"]
    g.VectorScaleMode = "Scale by Magnitude"
    g.ScaleFactor = 0.05
    g.GlyphMode = "All Points"
    gd = Show(g, view)
    ColorBy(gd, ("POINTS", None))
    gd.AmbientColor = EDGE
    gd.DiffuseColor = EDGE
    finish(view, "darcy_flow", twod=True, zoom=1.8, res=[1600, 620])
