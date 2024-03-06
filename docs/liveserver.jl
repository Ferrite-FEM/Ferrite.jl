#!/usr/bin/env julia

# Root of the repository
const repo_root = dirname(@__DIR__)

# Make sure docs environment is active and instantiated
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Communicate with docs/make.jl that we are running in live mode
push!(ARGS, "liveserver")

# Run LiveServer.servedocs(...)
import LiveServer
LiveServer.servedocs(;
    # Documentation root where make.jl and src/ are located
    foldername = joinpath(repo_root, "docs"),
    # Extra source folder to watch for changes
    include_dirs = [
        # Watch the src and ext folder so docstrings can be Revise'd
        joinpath(repo_root, "src"),
        joinpath(repo_root, "ext"),
    ],
    skip_dirs = [
        # Skip the folder where Literate.jl output is written. This is needed
        # to avoid infinite loops where running make.jl updates watched files,
        # which then triggers a new run of make.jl etc.
        joinpath(repo_root, "docs/src/tutorials"),
        joinpath(repo_root, "docs/src/howto"),
        joinpath(repo_root, "docs/src/gallery"),
    ],
    include_files = [
        joinpath(repo_root, "docs/generate.jl"),
        joinpath(repo_root, "docs/changelog.jl"),
        joinpath(repo_root, "CHANGELOG.md"),
        # Watch the index files in the skip_dirs folders
        joinpath(repo_root, "docs/src/tutorials/index.md"),
        joinpath(repo_root, "docs/src/howto/index.md"),
        joinpath(repo_root, "docs/src/gallery/index.md"),
    ],
    skip_files = [
        joinpath(repo_root, "docs/src/changelog.md"),
    ],
)
