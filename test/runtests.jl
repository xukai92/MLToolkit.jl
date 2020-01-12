using Test
using MLToolkit: include_list_as_module

@testset "Tests" begin
    tests = [
        "Plots/Plots.jl",    
        "Neural/Neural.jl",
        # "Datasets/Datasets.jl",
        "Scripting/Scripting.jl",
        "MonteCarlo/MonteCarlo.jl",
        # "distributions/distributions.jl",
        "utility.jl",
        # "special.jl",
        "transformations.jl",
    ]

    include_list_as_module(tests, "Test")
end
