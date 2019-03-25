using Test
using MLToolkit: include_list_as_module

@testset "Tests" begin
    tests = [
        "Data/Data",
        "neural/neural",
        "special",
        "scripting",
        "MonteCarlo/MonteCarlo",
        "distributions/distributions",
        "transformations",
    ]

    include_list_as_module(tests, "Test")
end
