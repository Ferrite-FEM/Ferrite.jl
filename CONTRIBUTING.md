# Ferrite.jl contributor guide

Welcome to Ferrite.jl contributor documentation! In this document you find
information about:

 - [Documentation](#documentation)
 - [Reporting issues](#reporting-issues)
 - [Code changes](#reporting-issues)

If you are new to open source development in general there are many guides online to help
you get started, for example [first-contributions][first-contributions]. Another great
resource, which specifically discusses Julia contributions, is the video [Open source, Julia
packages, git, and GitHub][tim-git].

## Documentation

Contributing to documentation is a great way to get started with any new project. As a new
user you have a unique perspective of what things need to be documented and explained better
-- if something confuses you, chances are you're not alone. Remember that also simple
changes like fixing typos are welcome contributions. If you are looking for specific things
to work on you can look at [open issues][open-issues].

Small changes can be done easily in GitHub's web interface (see [Editing
files][gh-edit-files]). Every page in the documentation have an `Edit on GitHub` button at
the top, which takes you to the correct source file. The video [Making Julia documentation
better][tim-doc] guides you through these steps.

Making larger changes is easier locally after cloning the repository. With a local
repository you can also preview the changes in your own browser. After starting a Julia REPL
in the root of the repository you can execute the following snippet:

```julia
include("docs/liveserver.jl")
```

This uses [`LiveServer.jl`][liveserver] to launch a local webserver which you can visit at
[http://localhost:8000](http://localhost:8000). `LiveServer.jl` will monitor changes to the
source files. When you make a change, and save the file, `LiveServer.jl` will automatically
rebuild the documentation, and automatically refresh your browser window. Note that the
first build may take longer, but subsequent runs should be substantially faster and give you
almost instant feedback in the browser.

**Useful resources**
 - General information about documenting Julia code in the [Julia manual][julia-doc].
 - [Documentation for `Documenter.jl`][documenter] which is used to render the HTML pages.
 - [Documentation for `Literate.jl`][literate] which is used for tutorials/examples.


## Reporting issues

If you have found a bug or a problem with Ferrite.jl you can open an [issue][new-issue]. Try
to include as much information about the problem as possible and preferably some code that
can be copy-pasted to reproduce it (see [How to create a Minimal, Reproducible
Example][so-mre]).

If you can identify a fix for the bug you can submit a pull request without first opening an
issue, see [Code changes](#code-changes).


## Code changes

Bug fixes and improvements to the code, or to the unit tests are always welcome. If you have
ideas about new features or functionality it might be good to first open an
[issue][new-issue] or [discussion][new-discussion] to get feedback before spending too much
time implementing something.

If you are looking for specific things to work on you can look at [open
issues][open-issues].

Remember to always include (when applicable): i) unit tests which exercises the new code,
ii) documentation, iii) a note in the [CHANGELOG.md](CHANGELOG.md) file.


[documenter]: https://juliadocs.github.io/Documenter.jl/
[first-contributions]: https://github.com/firstcontributions/first-contributions
[gh-edit-files]: https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files#editing-files-in-another-users-repository
[julia-doc]: https://docs.julialang.org/en/v1/manual/documentation/
[literate]: https://fredrikekre.github.io/Literate.jl/v2/
[liveserver]: https://github.com/tlienart/LiveServer.jl
[new-discussion]: https://github.com/Ferrite-FEM/Ferrite.jl/discussions/new
[new-issue]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/new
[open-issues]: https://github.com/Ferrite-FEM/Ferrite.jl/issues
[so-mre]: https://stackoverflow.com/help/minimal-reproducible-example
[tim-doc]: https://youtu.be/ZpH1ry8qqfw
[tim-git]: https://youtu.be/cquJ9kPkwR8
