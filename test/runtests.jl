using Distributed, Test

@testset "Tests" begin
    tests = [
        "Data/Data",
        "neural",
        "special",
        "scripting",
        "MonteCarlo/MonteCarlo",
        "distributions",
        "transformations",
    ]

    res = map(tests) do t
        @eval module $(Symbol("Test_", t))
            include($t * ".jl")
        end
        return
    end
end
