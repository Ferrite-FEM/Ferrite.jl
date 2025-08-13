# Papers using Ferrite
Ferrite has been used in a number of scientific publications, as shown by the reference list below.
If you are using Ferrite when preparing a manuscript, please cite Ferrite according to
[CITATION.cff](https://github.com/Ferrite-FEM/Ferrite.jl/blob/master/CITATION.cff)

After publication, please open a pull request to add your paper to
[ferritepapers.bib](https://github.com/Ferrite-FEM/Ferrite.jl/blob/master/docs/src/assets/ferritepapers.bib),
which will make it appear in the list below. If you are unsure how to do that, you can also add your bib-entry
to the following [issue](https://github.com/Ferrite-FEM/Ferrite.jl/issues/1205).

```@eval
using Bibliography, DocumenterCitations, Markdown

entries = Bibliography.import_bibtex("assets/ferritepapers.bib")
entries_by_year = Dict{Int, Vector}()
for entry in values(entries)
    year = parse(Int, entry.date.year)
    thisyears = get!(entries_by_year, year, Any[])
    push!(thisyears, entry)
end
bibparse(entry) = DocumenterCitations.format_bibliography_reference(:authoryear, entry)
years = reverse(sort(collect(keys(entries_by_year))))
content = prod("## $(year) \n" * join([bibparse(entry) for entry in entries_by_year[year]], "\n\n") * "\n\n" for year in years)
Markdown.parse(content)
```
