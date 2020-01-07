using Test
using MLToolkit: include_list_as_module

@testset "Tests" begin
    tests = [
        "Data/Data.jl",
        "Neural/Neural.jl",
        "utility.jl",
        # "special.jl",
        "Scripting/Scripting.jl",
        "MonteCarlo/MonteCarlo.jl",
        # "distributions/distributions.jl",
        "transformations.jl",
    ]

    include_list_as_module(tests, "Test")
end
