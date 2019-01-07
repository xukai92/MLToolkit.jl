using Distributed, Test

@testset "Tests" begin
    tests = [
        "data",
        "special",
        "activations",
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
