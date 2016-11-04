# Only run docs build from linux nightly build on travis.
get(ENV, "TRAVIS_OS_NAME", "")       == "linux"   || exit()
get(ENV, "TRAVIS_JULIA_VERSION", "") == "nightly" || exit()

Pkg.add("Documenter")
Pkg.checkout("Documenter")
include("make.jl")
