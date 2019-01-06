using MLToolkit, Test
using Distributions: Beta, Dirichlet
using Statistics: mean
using StatsFuns: logit, logistic

@testset "Gumbel" begin
    τ_atol_ratio = 2
    n = 100_000

    @testset "GumbelSoftmax" begin
        for _ = 1:NUM_RANDTESTS
            p = Matrix{FT}(rand(Dirichlet([1.0, 1.0]), 1))

            gs = GumbelSoftmax{AT}(p)
            x = hcat([rand(gs) for _ = 1:n]...)

            @test mean(x; dims=2) ≈ p atol=(2 * τ_atol_ratio * ATOL_RAND)
        end
    end

    @testset "GumbelBernoulli" begin
        for _ = 1:NUM_RANDTESTS
            p = Matrix{FT}(rand(Beta(1.0, 1.0), 1, 1))

            gs2d = GumbelSoftmax2D{AT}(p)
            gb = GumbelBernoulli{AT}(p)

            x = Array(hcat([rand(gs2d)[1,:] for _ = 1:n]...))
            @test mean(x; dims=2) ≈ p atol=ATOL_RAND

            x = Array([rand(gb)[1,1] for _ = 1:n])
            @test mean(x; dims=1) ≈ p atol=ATOL_RAND

        end
    end

    @testset "GumbelBernoulliLogit" begin
        for _ = 1:NUM_RANDTESTS
            p = Matrix{FT}(rand(Beta(1.0, 1.0), 1, 1))

            gbl = GumbelBernoulliLogit{AT}(logit.(p))

            logitx = Array([rand(gbl)[1] for _ = 1:n])
            x = logistic.(logitx)
            @test mean(x; dims=1) ≈ p atol=ATOL_RAND
        end
    end
end
