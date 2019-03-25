using Test
using MLToolkit: include_list_as_module

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

    include_list_as_module(tests, "TestDistributions")
end
