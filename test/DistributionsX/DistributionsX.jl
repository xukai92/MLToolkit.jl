using Test
using MLToolkit: include_list_as_module

@testset "DistributionsX" begin
    tests = [
        "noise.jl",
        "gumbel.jl",
        "bernoulli.jl",
        # "displaced_poisson.jl",
        # "ibp.jl",
        # "normal.jl",
        # "beta.jl",
        # "npd.jl",
    ]

    include_list_as_module(tests, "TestDistributionsX")

    @warn "`DistributionsX.jl` is not tested."
end
