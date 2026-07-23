# [Documentation](@id devdocs-documentation)

```@docs
Ferrite.asset_url
```

## Example figures and animations

The figures and animations shown on the [tutorials](../tutorials/index.md) and [code
gallery](../gallery/index.md) overview pages, and at the top of each example, are rendered
with [ParaView](https://www.paraview.org/) from the VTK files that the examples themselves
write. They come in light/dark pairs (`<name>-light.png` / `<name>-dark.png`, or `.webp`
animations for time-stepping examples) and `docs/src/assets/custom.css` displays the
variant matching the active Documenter theme.

The pipeline consists of the following pieces:

- `docs/generate_screenshots.jl` runs each example registered in its `EXAMPLES`
  dictionary, collecting the `.vtu`/`.pvd` output in `docs/screenshot-data/` (gitignored),
  and then invokes `pvbatch` on `docs/screenshots.py`.
- `docs/screenshots.py` contains one small ParaView scene per example and renders the
  light/dark pair for each into `docs/screenshot-assets/` (gitignored). Static scenes are
  saved with `finish(...)` (PNG) and time series with `finish_anim(...)` (animated WebP,
  requires ImageMagick's `magick` on `PATH`).
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
   animation, a paraview collection (`WriteVTK.paraview_collection`) with one file per
   time step. This output is what the figure is rendered from.
2. Register the example in the `EXAMPLES` dictionary in `docs/generate_screenshots.jl`,
   mapping a screenshot name to the literate source file.
3. Add a scene for it in `docs/screenshots.py` (copying a similar existing scene is the
   easiest way; use `finish` for a static PNG and `finish_anim` for an animation).
4. Render it with `julia --project=docs docs/generate_screenshots.jl <name>` (requires
   `pvbatch` on `PATH`). While tuning the scene, add `--render-only` to reuse the data
   files from the previous run. Check the result in `docs/screenshot-assets/`.
5. Register the generated file names in `docs/download_resources.jl` so that CI (and other
   machines) fetch them when building the docs.
6. Show the figure at the top of the example by adding an `# ![](<name>-light.png)` /
   `# ![](<name>-dark.png)` pair after the title in the literate source file (see e.g.
   `docs/src/literate-tutorials/heat_equation.jl`).
7. Add the example to the overview page: an entry in the corresponding `cards` list in
   `docs/generate.jl` *and* a `---`-separated description section in
   `docs/tutorials_index_body.md` or `docs/gallery_index_body.md`. The cards and the
   description sections are paired in order, so put both in the same position (the build
   errors if the counts don't match).
8. Verify with a local docs build (`julia --project=docs docs/make.jl`), which picks up
   the assets from `docs/screenshot-assets/`.
9. Once happy, upload the assets to `gh-pages` with
   `julia --project=docs docs/generate_screenshots.jl --render-only <name> --upload`
   (requires push rights to the repository).
