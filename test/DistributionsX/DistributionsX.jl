using Test, MLToolkit.DistributionsX

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
    foreach(include, tests)

    @warn "`DistributionsX.jl` is not tested."
end
