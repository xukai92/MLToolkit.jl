using Test
using MLToolkit: include_list_as_module

@testset "Distributions" begin
    tests = [
        "displaced_poisson.jl",
        "ibp.jl",
        "normal.jl",
        "bernoulli.jl",
        "gumbel.jl",
        "beta.jl",
        "npd.jl",
    ]

    include_list_as_module(tests, "TestDistributions")
end
