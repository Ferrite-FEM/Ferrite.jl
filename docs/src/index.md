```@meta
DocTestSetup = :(using Ferrite)
```

# Ferrite.jl

Welcome to the documentation for Ferrite.jl! Ferrite is a finite element toolbox that
provides functionalities to implement finite element analysis in
[Julia](https://github.com/JuliaLang/julia). The aim is to be i) general, ii) performant,
and iii) to keep mathematical abstractions.

!!! note "Upgrading code from version 0.3.x to version 1.0"
    Ferrite version 1.0 contains a number of breaking changes compared to version 0.3.x. The
    [Changelog](changelog.md) documents all changes and there is also a section specifically
    for [Upgrading code from Ferrite 0.3 to 1.0](@ref).

!!! note
    Please help improve this documentation -- if something confuses you, chances are you're
    not alone. It's easy to do as you read along: just click on the "Edit on GitHub" link at
    the top of each page, and then [edit the files directly in your browser]
    (https://help.github.com/articles/editing-files-in-another-user-s-repository/). Your
    changes will be vetted by developers before becoming permanent, so don't worry about
    whether you might say something wrong. See also [Contributing to Ferrite](@ref) for more
    details.

## How the documentation is organized

This high level view of the documentation structure will help you find what you are looking
for. The document is organized as follows[^1]:

 - [**Tutorials**](tutorials/index.md) are thoroughly documented examples which guides you
   through the process of solving partial differential equations using Ferrite.
 - [**Topic guides**](topics/index.md) contains more in-depth explanations and discussions
   about finite element programming concepts and ideas, and specifically how these are
   realized in Ferrite.
 - [**Reference**](reference/index.md) contains the technical API reference of functions and
   methods (e.g. the documentation strings).
 - [**How-to guides**](howto/index.md) will guide you through the steps involved in
   addressing common tasks and use-cases. These usually build on top of the tutorials and
   thus assume basic knowledge of how Ferrite works.

[^1]: The organization of the document follows the [Diátaxis Framework](https://diataxis.fr).

The four sections above form the main user-facing parts of the documentation. In addition,
the document also contain the following sections:

 - [**Code gallery**](gallery/index.md) contains user contributed example programs showcasing
   what can be done with Ferrite.
 - [**Changelog**](changelog.md) contains release notes and information about how to upgrade
   between releases.
 - [**Developer documentation**](devdocs/index.md) contains documentation of Ferrite internal
   code and is mainly targeted at developers of Ferrite.

## Getting started

As a new user of Ferrite it is suggested to start working with the tutorials before using
Ferrite to tackle the specific equation you ultimately want to solve. The tutorials start
with explaining the basic concepts and then increase in complexity. Understanding the first
tutorial program, [solving the heat equation](@ref tutorial-heat-equation), is essential in order to
understand how Ferrite works. Already this rather simple program discusses many of the
important concepts. See the [tutorials overview](@ref Tutorials) for suggestion on how to
progress to more advanced usage.

### Getting help

If you have questions about Ferrite it is suggested to use the `#ferrite-fem` channel on the
[Julia Slack](https://julialang.org/slack/), or the `#Ferrite.jl` stream on
[Zulip](https://julialang.zulipchat.com/). Alternatively you can use the [discussion
forum](https://github.com/Ferrite-FEM/Ferrite.jl/discussions) on the GitHub repository.

### Installation

To use Ferrite you first need to install Julia, see <https://julialang.org/> for details.
Installing Ferrite can then be done from the Pkg REPL; press `]` at the `julia>` prompt to
enter `pkg>` mode:

```
pkg> add Ferrite
```

This will install Ferrite and all necessary dependencies. Press backspace to get back to the
`julia>` prompt. (See the [documentation for Pkg](https://pkgdocs.julialang.org/), Julia's
package manager, for more help regarding package installation and project management.)

Finally, to load Ferrite, use

```julia
using Ferrite
```

You are now all set to start using Ferrite!


## Contributing to Ferrite

Ferrite is still under active development. If you find a bug, or have ideas for
improvements, you are encouraged to interact with the developers on the [Ferrite GitHub
repository](https://github.com/Ferrite-FEM/Ferrite.jl). There is also a thorough contributor
guide which can be found in
[CONTRIBUTING.md](https://github.com/Ferrite-FEM/Ferrite.jl/blob/master/CONTRIBUTING.md).
