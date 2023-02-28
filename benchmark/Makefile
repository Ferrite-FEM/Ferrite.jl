# Allow custom julia commands
JULIA_CMD ?= julia

BENCHMARK_DIR := $(shell dirname $(abspath $(firstword $(MAKEFILE_LIST))))

default: benchmark

setup-dev:
	${JULIA_CMD} --project=${BENCHMARK_DIR} -e 'using Pkg; Pkg.develop(path=".."); Pkg.instantiate(); Pkg.precompile();'

benchmark: setup-dev
	${JULIA_CMD} --project=${BENCHMARK_DIR} ${BENCHMARK_DIR}/runbenchmarks.jl benchmark/benchmarks-main.jl

benchmark-%: setup-dev
	${JULIA_CMD} --project=${BENCHMARK_DIR} ${BENCHMARK_DIR}/runbenchmarks.jl benchmark/${@}-main.jl

compare: setup-dev
	${JULIA_CMD} --project=${BENCHMARK_DIR} ${BENCHMARK_DIR}/runcomparison.jl ${target} ${baseline} benchmark/benchmarks-main.jl

compare-%: setup-dev
	${JULIA_CMD} --project=${BENCHMARK_DIR} ${BENCHMARK_DIR}/runcomparison.jl ${target} ${baseline} "benchmark/benchmark-${@:compare-%=%}-main.jl"

.PHONY: default benchmark benchmark-% compare compare-%
