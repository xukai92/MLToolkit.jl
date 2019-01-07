using Distributed, Test

@testset "Tests" begin
    tests = [
        "data",
        "neural",
        "special",
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
