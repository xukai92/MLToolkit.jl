using Distributed, Test
using MLToolkit.MonteCarlo

@testset "Monte Carlo" begin

    tests = [
        "russian_roulette",
    ]

    res = map(tests) do t
        @eval module $(Symbol("TestMonteCarlo_", t))
            include($t * ".jl")
        end
        return
    end

    @testset "estimator_stats" begin
        n_mc = 10_000
        estimates = zeros(n_mc)
        times = zeros(n_mc)
        ground = 0.0
        for i = 1:10_000
            t = @elapsed e = randn()
            estimates[i] = e
            times[i] = t
        end
        stats = estimator_stats(estimates; times=times, ground=ground)
    end
    
end