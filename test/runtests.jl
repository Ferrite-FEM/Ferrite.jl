using Ferrite
using ParallelTestRunner

const TESTDIR = @__DIR__

using Test

if isdefined(Test, :detect_closure_boxes)
    @test isempty(Test.detect_closure_boxes(Ferrite))
end

# `find_tests` auto-discovers every `.jl` file in `test/` (recursively). Each is
# run in its own isolated worker process, so files must be self-contained: they
# carry their own `using`/`import` and `include("test_utils.jl")` when needed.
# The only common setup injected into every test is `using Ferrite`.
testsuite = find_tests(TESTDIR)

# Drop files that are discovered but are not standalone test files to run here:
for name in (
        "test_utils",       # shared helpers, `include`d by the tests that need them
        "interpolation_test_utils",           # shared helpers for the test_interpolations* files
        "integration/convergence_test_utils", # shared helpers for the integration convergence tests
        "jet",              # JET tests, run separately
        "test_notebooks",   # notebook tests, opt-in
        "coverage/coverage", # coverage tooling, not a test
    )
    delete!(testsuite, name)
end
# GPU tests run separately on Buildkite (see test/GPU/runtests.jl). The CPU-backend
# variant is `test_ka_cpu.jl`, which includes the shared files from test/GPU itself.
filter!(((name, _),) -> !startswith(name, "GPU/"), testsuite)

# Auto CPU thread count detection in ParallelTestRunner is bad
push!(ARGS, "--jobs=$(Sys.CPU_THREADS)")

# `init_code` runs in each test's (isolated) sandbox module; `using Ferrite` is
# the only setup common to all tests. `init_worker_code` runs once per worker in
# `Main`: loading these there makes type names print unqualified (e.g.
# `Lagrange`, `SparseMatrixCSC`), matching a normal `using Ferrite` session,
# which the `show`/`repr` tests rely on.
runtests(
    Ferrite, ARGS;
    testsuite,
    init_code = :(using Ferrite),
    init_worker_code = :(using Ferrite, LinearAlgebra, SparseArrays, SparseMatricesCSR),
)
