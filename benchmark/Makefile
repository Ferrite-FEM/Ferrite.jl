# Allow custom julia commands
JULIA_CMD ?= julia

BENCHMARK_DIR := $(shell dirname $(abspath $(firstword $(MAKEFILE_LIST))))

default: benchmark

benchmark:
	${JULIA_CMD} --project=${BENCHMARK_DIR} ${BENCHMARK_DIR}/runbenchmarks.jl

compare:
	${JULIA_CMD} --project=${BENCHMARK_DIR} ${BENCHMARK_DIR}/runcomparison.jl ${target} ${baseline}

.PHONY: default benchmark benchmark compare
