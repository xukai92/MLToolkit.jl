using Test, Distributed

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

    pmap(tests) do t
        @eval module $(Symbol("Test", t))
            include($t)
        end
        return
    end
end
