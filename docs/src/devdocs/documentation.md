# [Documentation](@id devdocs-documentation)

```@docs
Ferrite.asset_url
```

## Example figures and animations

The figures and animations shown on the [tutorials](../tutorials/index.md) and [code
gallery](../gallery/index.md) overview pages, in the [how-to guides](../howto/index.md),
and at the top of each example, are rendered with
[ParaView](https://www.paraview.org/) from the VTK files that the examples themselves
write. They come in light/dark pairs (`<name>-light.png` / `<name>-dark.png`, or `.webp`
animations for time-stepping examples) and `docs/src/assets/custom.css` displays the
variant matching the active Documenter theme.

The pipeline consists of the following pieces:

- `docs/generate_screenshots.jl` runs each example registered in its `EXAMPLES`
  dictionary, collecting its `.vtu`/`.pvd`/`.vtkhdf` output in
  `docs/screenshot-data/` (gitignored),
  and then invokes `pvbatch` on `docs/screenshots.py`.
- `docs/screenshots.py` contains one small ParaView scene per example and renders the
  light/dark pair for each into `docs/screenshot-assets/` (gitignored). Static scenes are
  saved with `finish(...)` (PNG) and time series with `finish_anim(...)` (animated WebP,
  requires ImageMagick's `magick` or `convert` on `PATH`).
- The rendered files are *not* committed to the main branch. They are uploaded to the
  `assets/` directory on the `gh-pages` branch (pass `--upload` to
  `generate_screenshots.jl`), from where `docs/download_resources.jl` fetches them during
  the docs build. When previewing locally, `download_resources.jl` prefers freshly
  rendered files in `docs/screenshot-assets/` over downloading, so no upload is needed.
- `docs/generate.jl` (`write_overview`) composes the overview pages from the curated
  descriptions in `docs/tutorials_index_body.md` and `docs/gallery_index_body.md`: each
  `---`-separated description section is paired, in order, with an entry in the `cards`
  lists and rendered as a row with the figure to the left of the description.

### Adding a figure for a new example

1. Make sure the example writes its result to a VTK file (`VTKGridFile`), or, for an
   animation, a temporal `VTKHDFGridFile` (or a `WriteVTK.paraview_collection` with one
   file per time step). This output is what the figure is rendered from. If the figure
   needs data the example does not write by default — a second parameter set to compare
   against, a finer mesh — put that extra call in the `POSTRUN` dictionary instead of
   making the example itself slower for every docs build.
2. Register the example in the `EXAMPLES` dictionary in `docs/generate_screenshots.jl`,
   mapping a screenshot name to the literate source file.
3. Add a scene for it in `docs/screenshots.py` (copying a similar existing scene is the
   easiest way; use `finish` for a static PNG and `finish_anim` for an animation).
   Anything drawn on top of the data rather than coloured by it — glyphs, a marked-up cut
   line — should go through `annotate`, so its colour flips with the variant instead of
   disappearing into one of the two backgrounds. A scene that renders more than one figure
   also needs an `OUTPUTS` entry in `docs/generate_screenshots.jl` listing its file
   basenames.
4. Render it with `julia --project=docs docs/generate_screenshots.jl <name>` (requires
   `pvbatch` on `PATH`, and `matplotlib` importable by it — `screenshots.py` says how if
   it isn't). While tuning the scene, add `--render-only` to reuse the data files from the
   previous run. Check the result in `docs/screenshot-assets/`.
5. Register the generated file names in `docs/download_resources.jl` so that CI (and other
   machines) fetch them when building the docs.
6. Show the figure at the top of the example by adding an `# ![](<name>-light.png)` /
   `# ![](<name>-dark.png)` pair after the title in the literate source file (see e.g.
   `docs/src/literate-tutorials/heat_equation.jl`). Give the caption a named anchor,
   `# [[*Figure 1*](@ref <page-id>-figure-1)](@id <page-id>-figure-1):` on its own line
   followed by the caption text, where `<page-id>` is the `@id` of the page title. The
   inner `@ref` makes the label a link to itself (so its URL can be copied), and the
   figure can be referenced from the text (on any page) with
   `[Figure 1](@ref <page-id>-figure-1)`.
7. For a tutorial or gallery example, add it to the overview page: an entry in the
   corresponding `cards` list in `docs/generate.jl` *and* a `---`-separated description
   section in
   `docs/tutorials_index_body.md` or `docs/gallery_index_body.md`. The cards and the
   description sections are paired in order, so put both in the same position (the build
   errors if the counts don't match).
8. Verify with a local docs build (`julia --project=docs docs/make.jl`), which picks up
   the assets from `docs/screenshot-assets/`.
9. Once happy, upload the assets to `gh-pages` with
   `julia --project=docs docs/generate_screenshots.jl --render-only <name> --upload`
   (requires push rights to the repository).

The Documenter CI job also runs every scene in `--check` mode. It reuses the VTK output
from notebook execution, renders at reduced resolution with only a few animation frames,
and never uploads those smoke-test assets.

## Illustrations in the topic guides

The drawn figures — reference shape numbering, grid numbering, the geometric mapping, and
`Dirichlet` vs `ProjectedDirichlet` — are generated by `docs/diagrams.jl`, which writes a
light/dark SVG pair per figure into `docs/src/topics/assets/`. It is plain Julia emitting
SVG, with no dependencies, and one geometry definition per figure shared by both color
themes so the two variants cannot drift apart.

Unlike the example figures these are small enough to commit, so they live in the repository
and only need regenerating when a figure changes, with
`julia docs/diagrams.jl docs/src/topics/assets`.

Prefer explaining notation in the page over lettering it into the figure: text in the
markdown is selectable, searchable, and typeset by KaTeX, and it does not need a second
color variant.

## Element atlas

The [element atlas](@ref element-atlas) presents one representative example per
family shipped by Ferrite. It is generated during
`docs/make.jl`; the generated Markdown and SVGs in `docs/src/elements/` are gitignored,
like the generated tutorials.

- `docs/element_atlas/catalog.jl` describes the families and their examples.
- `docs/element_atlas/generate.jl` evaluates Ferrite's basis functions and the explicit
  dof functionals, verifies duality, and writes the pages and SVGs. `functionals.jl`
  specifies moment weights and orientations: the `DofFunctional` types alone do not
  encode these details.
- `docs/element_atlas/render.jl` draws scalar surfaces, vector fields, edge moment
  densities as standalone, accessible SVGs. The geometry is shared
  by the light and dark palettes. No plotting dependency or external asset is needed.
- `docs/src/assets/element-atlas.css` styles the section; `element-atlas.js` synchronizes
  the basis selector with clickable dof markers. The markers use native buttons
  positioned from the SVG renderer's coordinates.
  They support Enter and Space and retain focus when the basis changes. Downloaded SVGs
  remain standalone figures. With JavaScript disabled, every basis is visible.
  Math uses Documenter's existing KaTeX renderer.

To regenerate only the atlas and run its mathematical checks, from the repository root:

```sh
jld --project=docs --idle-timeout=2h eval 'include("docs/element_atlas/generate.jl"); ElementAtlas.generate()'
```

To preview it within the complete documentation without executing the tutorials or
deploying, build in the existing draft mode:

```sh
jld --project=docs --idle-timeout=2h eval 'push!(ARGS, "liveserver"); include("docs/make.jl")'
python3 -m http.server --directory docs/build 8000
```

Then open `http://localhost:8000/elements/`. A normal docs build generates the same
atlas. `ElementAtlas.generate(; prettyurls = false)` also supports a flat-URL
Documenter build; this setting must agree with `Documenter.HTML(prettyurls = ...)`
because the figures and download links use raw HTML.

To extend the catalogue, add a representative example in `catalog.jl`, specify its
space and functionals including weights and orientation, and choose a visualization
suitable for those functionals. Every example must pass the complete duality check
before generation. Do not infer moment weights from `NormalMoment` or
`TangentialMoment` alone. Review new figures in both themes and compare displayed
dof ordering against Ferrite's entity ordering.
