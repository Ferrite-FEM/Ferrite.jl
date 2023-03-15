# Allow custom julia commands
JULIA_CMD ?= julia

BENCHMARK_DIR := $(shell dirname $(abspath $(firstword $(MAKEFILE_LIST))))

default: benchmark

clean:
	rm Manifest.toml

Manifest.toml:
	${JULIA_CMD} --project=${BENCHMARK_DIR} -e 'using Pkg; Pkg.develop(path=".."); Pkg.instantiate(); Pkg.precompile();'

benchmark: Manifest.toml
	${JULIA_CMD} --project=${BENCHMARK_DIR} ${BENCHMARK_DIR}/runbenchmarks.jl benchmark/benchmarks.jl

benchmark-%: Manifest.toml
	${JULIA_CMD} --project=${BENCHMARK_DIR} ${BENCHMARK_DIR}/runbenchmarks.jl benchmark/benchmarks.jl ${@:benchmark-%=%}

compare: Manifest.toml
	${JULIA_CMD} --project=${BENCHMARK_DIR} ${BENCHMARK_DIR}/runcomparison.jl ${target} ${baseline} benchmark/benchmarks.jl

compare-%: Manifest.toml
	${JULIA_CMD} --project=${BENCHMARK_DIR} ${BENCHMARK_DIR}/runcomparison.jl ${target} ${baseline} benchmark/benchmark.jl ${@:benchmark-%=%}

.PHONY: default clean benchmark benchmark-% compare compare-%
