using Test, MLToolkit
using Distributions: Beta, Dirichlet
using Statistics: mean
using StatsFuns: logit, logistic

@testset "Gumbel" begin
    τ_atol_ratio = 2
    n = 100_000

    @testset "BatchGumbelSoftmax" begin
        for _ = 1:NUM_RANDTESTS
            p = Matrix{FT}(rand(Dirichlet([1.0, 1.0]), 1))

            gs = BatchGumbelSoftmax(AT(p))
            x = hcat([rand(gs) for _ = 1:n]...)

            @test mean(x; dims=2) ≈ p atol=(2 * τ_atol_ratio * ATOL_RAND)
            @test mean(x; dims=2) ≈ mean(gs) atol=(2 * τ_atol_ratio * ATOL_RAND)
        end
    end

    @testset "BatchGumbelBernoulli" begin
        for _ = 1:NUM_RANDTESTS
            p = Matrix{FT}(rand(Beta(1.0, 1.0), 1, 1))

            gs2d = BatchGumbelSoftmax2D(AT(p))
            gb = BatchGumbelBernoulli(AT(p))

            x = Array(hcat([rand(gs2d)[1,:] for _ = 1:n]...))
            # @info "" x p rand(gs2d)
            @test mean(x; dims=2) ≈ p atol=ATOL_RAND

            x = Array([rand(gb)[1,1] for _ = 1:n])
            @test mean(x; dims=1) ≈ p atol=ATOL_RAND
            @test mean(x; dims=1) ≈ Array(mean(gb)) atol=ATOL_RAND

            x = rand(gb)
            @test logpdf(gb, x) ≈ logpdfCoV(gb, x) atol=100ATOL
        end
    end

    @testset "BatchGumbelBernoulliLogit" begin
        for _ = 1:NUM_RANDTESTS
            p = Matrix{FT}(rand(Beta(1.0, 1.0), 1, 1))

            gbl = BatchGumbelBernoulliLogit(AT(logit.(p)))

            logitx = Array([logitrand(gbl)[1] for _ = 1:n])
            x = logistic.(logitx)
            @test mean(x; dims=1) ≈ p atol=ATOL_RAND
            @test mean(x; dims=1) ≈ Array(mean(gbl)) atol=ATOL_RAND
        end

        @warn "`logpdflogit(gbl::BatchGumbelBernoulliLogit, logitx)` is not tested."
    end
end
