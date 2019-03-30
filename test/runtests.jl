using Test
using MLToolkit: include_list_as_module

@testset "Tests" begin
    tests = [
        "Data/Data.jl",
        "neural/neural.jl",
        "special.jl",
        "scripting.jl",
        "MonteCarlo/MonteCarlo.jl",
        "distributions/distributions.jl",
        "transformations.jl",
    ]

    include_list_as_module(tests, "Test")
end
