using Test, Distributed

@testset "Tests" begin
    tests = [
        # Cleaned
        "Plots/Plots.jl",
        "Datasets/Datasets.jl",
        "Scripting/Scripting.jl",
        # To clean
        "Neural/Neural.jl",
        "MonteCarlo/MonteCarlo.jl",
        "utility.jl",
        # "special.jl",
        "transformations.jl",
        # Deprecated
        # "distributions/distributions.jl",
        # "neural/neural.jl",
    ]

    pmap(tests) do t
        @eval module $(Symbol("Test", t))
            include($t)
        end
        return
    end
end
