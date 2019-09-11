using Test
using MLToolkit: include_list_as_module

@testset "Ratio" begin
    tests = [
        "moment_matching.jl",
    ]

    include_list_as_module(tests, "TestRatio")
end