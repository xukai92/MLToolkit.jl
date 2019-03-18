using Distributed, Test

@testset "Distributions" begin
    tests = [
        "displaced_poisson",
        "ibp",
        "normal",
        "bernoulli",
        "gumbel",
        "beta",
        "npd",
    ]

    res = map(tests) do t
        @eval module $(Symbol("TestDistributions_", t))
            include("distributions/" * $t * ".jl")
        end
        return
    end
end
