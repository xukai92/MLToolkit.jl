using Test
using MLToolkit: include_list_as_module

@testset "DistributionsX" begin
    tests = [
        "noise.jl",
        "gumbel.jl",
    ]

    include_list_as_module(tests, "TestDistributionsX")

    @warn "`DistributionsX.jl` is not tested."
end
