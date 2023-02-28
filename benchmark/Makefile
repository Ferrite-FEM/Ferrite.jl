# Allow custom julia commands
JULIA_CMD ?= julia

BENCHMARK_DIR := $(shell dirname $(abspath $(firstword $(MAKEFILE_LIST))))

default: benchmark

setup-dev:
	${JULIA_CMD} --project=${BENCHMARK_DIR} -e 'using Pkg; Pkg.develop(path=".."); Pkg.instantiate(); Pkg.precompile();'

benchmark: setup-dev
	${JULIA_CMD} --project=${BENCHMARK_DIR} ${BENCHMARK_DIR}/runbenchmarks.jl

compare: setup-dev
	${JULIA_CMD} --project=${BENCHMARK_DIR} ${BENCHMARK_DIR}/runcomparison.jl ${target} ${baseline}

.PHONY: default benchmark benchmark compare
