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

datadir, outdir = sys.argv[1], sys.argv[2]
selected = sys.argv[3:]
RES = [1600, 1200]
ANIM_RES = [1000, 750]  # animations are rendered directly at their final size
EDGE = [0.15, 0.15, 0.15]  # mesh edge color (reads on both light and dark surfaces)
BARS = []  # scalar bars whose text flips between the variants
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
             horizontal=False, pos=None, fmt=None):
    ColorBy(display, array)
    display.SetScalarBarVisibility(view, True)
    lut = GetColorTransferFunction(array[1])
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
    else:
        # right of the mesh, vertically centred, with the title on top
        bar.Orientation = "Vertical"
        bar.HorizontalTitle = 1
        bar.WindowLocation = "Any Location"
        bar.Position = pos or [0.83, 0.32]
        bar.ScalarBarLength = 0.36
    bar.ScalarBarThickness = 24
    bar.TitleFontSize = 36
    bar.LabelFontSize = 32
    bar.TitleBold = 1
    bar.LabelBold = 1
    if fmt is not None:
        # e.g. "%.0f": plain numbers for the min/max labels where the default
        # exponent format grows too wide (the in-between labels stay automatic)
        bar.RangeLabelFormat = fmt
    BARS.append(bar)
    return lut


def _set_camera(view, azimuth, elevation, zoom, twod, pan_y=0.0):
    view.ResetCamera(False)
    cam = GetActiveCamera()
    if twod:
        view.CameraParallelProjection = 1
    else:
        cam.Azimuth(azimuth)
        cam.Elevation(elevation)
    view.ResetCamera(False)
    cam.Zoom(zoom)
    # pan_y < 0 shifts the scene up in the frame, freeing space for a
    # horizontal colour bar below the mesh
    cam.SetWindowCenter(0.0, pan_y)


def _apply_variant(view, text, bg):
    view.Background = bg
    for bar in BARS:
        bar.TitleColor = text
        bar.LabelColor = text


def finish(view, name, azimuth=30, elevation=25, zoom=1.0, twod=False, res=None,
           pan_y=0.0):
    # res overrides the frame size (default RES); use a matching aspect ratio for
    # non-square domains so the scene fills the frame instead of leaving margins.
    res = res or RES
    view.ViewSize = res
    Render()  # settle the render window / scalar-bar layout at the new size first
    _set_camera(view, azimuth, elevation, zoom, twod, pan_y)
    for variant, text, bg in VARIANTS:
        _apply_variant(view, text, bg)
        Render()
        SaveScreenshot(
            outdir + "/" + name + "-" + variant + ".png", view,
            ImageResolution=res, TransparentBackground=1,
        )
    BARS.clear()
    Delete(view)


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


def finish_anim(view, source, name, azimuth=30, elevation=25, zoom=1.0,
                twod=False, delay=8, res=None, pan_y=0.0):
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
    # Frame once at the last step (largest extent for deforming geometry).
    view.ViewTime = times[-1]
    Render()  # settle the render window / scalar-bar layout at the new size first
    _set_camera(view, azimuth, elevation, zoom, twod, pan_y)
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
                        title="$\\sigma_{22}$", horizontal=True)
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
    bar = GetScalarBar(lut_cell, view)
    bar.UseCustomLabels = 1  # the auto labels crowd together for this range
    bar.CustomLabels = [5000.0, 10000.0, 15000.0]
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
                     title="von Mises [Pa]", horizontal=True)
    pd = r.GetDataInformation().GetPointDataInformation()
    hi = max(pd.GetArrayInformation("σvM_dirichlet_3").GetComponentRange(0)[1],
             pd.GetArrayInformation("σvM_periodic_3").GetComponentRange(0)[1])
    lut_d.RescaleTransferFunction(0.0, hi)
    lut_p.RescaleTransferFunction(0.0, hi)
    bar = GetScalarBar(lut_p, view)
    bar.UseCustomLabels = 1  # the auto labels crowd together for this range
    bar.CustomLabels = [2.0e10, 4.0e10, 6.0e10]
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


# === animations (time-stepping examples) ===================================

# --- transient_heat_equation: temperature evolving on the unit square
@scene("transient_heat")
def scene_transient_heat():
    view = new_view()
    r = OpenDataFile(datadir + "/transient-heat.pvd")
    d = surface(r, view, edges=False)
    lut = colorbar(d, view, ("POINTS", "u"), title="T")
    lut.RescaleTransferFunction(*data_range_over_time(r, ("POINTS", "u")))
    finish_anim(view, r, "transient_heat", twod=True, zoom=0.95)


# --- porous_media: vertical strain (whole domain) and pressure evolution
@scene("porous_media")
def scene_porous_media():
    view = new_view()
    r = OpenDataFile(datadir + "/porous_media.pvd")
    # left panel: vertical strain from the displacement gradient; unlike the
    # pressure it is defined on the whole domain (also the solid parts)
    grad = Gradient(Input=r)
    grad.ScalarArray = ["POINTS", "u"]
    left = surface(grad, view, edges=False)
    lut_e = colorbar(left, view, ("POINTS", "Gradient"),
                     title="$\\epsilon_{22}$", pos=[0.02, 0.32], fmt="%.3f")
    lut_e.VectorMode = "Component"
    lut_e.VectorComponent = 4  # du_y/dy
    lut_e.RescaleTransferFunction(
        *data_range_over_time(grad, ("POINTS", "Gradient"), comp=4,
                              times=list(r.TimestepValues)))
    # right panel: pressure (NaN in the solid inclusions)
    shifted = Transform(Input=r)
    shifted.Transform.Translate = [6.5, 0.0, 0.0]
    right = surface(shifted, view, edges=False)
    lut_p = colorbar(right, view, ("POINTS", "p"), title="p", pos=[0.84, 0.32],
                     fmt="%.0f")
    lut_p.RescaleTransferFunction(*data_range_over_time(r, ("POINTS", "p")))
    lut_p.NanColor = [0.7, 0.7, 0.7]  # solid inclusions carry no pressure dof
    finish_anim(view, r, "porous_media", twod=True, zoom=0.92, res=[900, 750])


# --- ns_vs_diffeq: velocity magnitude, von Karman vortex street
@scene("ns_vs_diffeq")
def scene_ns_vs_diffeq():
    view = new_view()
    r = OpenDataFile(datadir + "/vortex-street.pvd")
    d = surface(r, view)
    lut = colorbar(d, view, ("POINTS", "v"), title="speed", horizontal=True)
    lut.RescaleTransferFunction(*data_range_over_time(r, ("POINTS", "v")))
    # wide channel domain -> wide frame, horizontal colour bar centred below the flow
    finish_anim(view, r, "ns_vs_diffeq", twod=True, zoom=1.7, res=[1146, 650],
                pan_y=-0.18)


# --- reactive_surface: reaction-diffusion pattern on a sphere
@scene("reactive_surface")
def scene_reactive_surface():
    view = new_view()
    r = OpenDataFile(datadir + "/reactive-surface.pvd")
    d = surface(r, view, edges=False)
    lut = colorbar(d, view, ("POINTS", "reactants"), title="reactant")
    lut.RescaleTransferFunction(*data_range_over_time(r, ("POINTS", "reactants")))
    finish_anim(view, r, "reactive_surface", azimuth=30, elevation=20, zoom=1.0)


# === code gallery ==========================================================

# --- helmholtz: solution of the Helmholtz equation on the unit square
@scene("helmholtz")
def scene_helmholtz():
    view = new_view()
    r = OpenDataFile(datadir + "/helmholtz.vtu")
    d = surface(r, view)
    colorbar(d, view, ("POINTS", "u"), title="u")
    finish(view, "helmholtz", twod=True, zoom=0.95)


# --- landau: Ginzburg-Landau polarisation, initial and minimised states
@scene("landau")
def scene_landau():
    for src, name in (("landauorig", "landau_orig"), ("landaufinal", "landau_opt")):
        view = new_view()
        r = OpenDataFile(datadir + "/" + src + ".vtu")
        d = surface(r, view, edges=False)
        colorbar(d, view, ("POINTS", "P"), title="$\\vert P \\vert$")
        finish(view, name, azimuth=30, elevation=35, zoom=1.0)


# --- topology_optimization: density evolution during optimisation (SIMP)
@scene("topology_optimization")
def scene_topology_optimization():
    view = new_view()
    r = open_series(datadir + "/topopt_frames_*.vtu")
    d = surface(r, view, edges=False)
    lut = colorbar(d, view, ("CELLS", "density"), title="density", horizontal=True)
    lut.RescaleTransferFunction(0.0, 1.0)  # SIMP density in [0, 1]
    # wide beam domain -> wide frame, horizontal colour bar centred below the mesh
    finish_anim(view, r, "topology_optimization", twod=True, zoom=1.4,
                res=[1200, 770], pan_y=-0.15)


# --- quasi_incompressible_hyperelasticity: deforming mixed u/p cube
@scene("quasi_incompressible_hyperelasticity")
def scene_quasi_incompressible_hyperelasticity():
    view = new_view()
    r = OpenDataFile(datadir + "/hyperelasticity_incomp_mixed.pvd")
    w = warp(r, "u", 1.0)
    d = surface(w, view, edges=False)
    lut = colorbar(d, view, ("POINTS", "u"), title="$\\vert u \\vert$")
    lut.RescaleTransferFunction(*data_range_over_time(r, ("POINTS", "u")))
    finish_anim(view, r, "quasi_incompressible_hyperelasticity",
               azimuth=35, elevation=20, zoom=1.1)


names = selected or list(SCENES)
unknown = [n for n in names if n not in SCENES]
if unknown:
    sys.exit("unknown scene(s): " + ", ".join(unknown))
for i, name in enumerate(names):
    print("[%d/%d] %s" % (i + 1, len(names), name), flush=True)
    SCENES[name]()

print("screenshots for", ", ".join(names), "written to", outdir)
