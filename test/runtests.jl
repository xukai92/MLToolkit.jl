using Distributed, Test

tests = [
    "data",
    "special",
    "activations",
    "distributions",
    "transformations",
]

@testset "Tests" begin
    res = map(tests) do t
        include("$t.jl")
        return
    end
end
