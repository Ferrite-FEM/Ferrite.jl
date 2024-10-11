using CUDA
using Ferrite
using Test

@test CUDA.functional()
generate_grid(Triangle, (2, 2))
