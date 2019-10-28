using Test, MLToolkit
using MLToolkit.DistributionsX: u2gumbel
using Distributions: Dirichlet
using Statistics: mean
using Tracker: gradient
using FiniteDifferences: central_fdm
# using Distributions: Beta
# using StatsFuns: logit, logistic

@testset "Gumbel" begin
    n = 5_000

    @testset "GumbelSoftmax" begin
        for _ = 1:NUM_RANDTESTS
            p = rand(Dirichlet([1.0, 1.0]))
            gs = GumbelSoftmax(p)

            for xs in [
                hcat([rand(gs) for _ = 1:n]...),
                rand(gs, n)
            ]
                @test mean(gs) == p
                @test vec(mean(xs; dims=2)) ≈ mean(gs) atol=2ATOL_RAND
            end

            p = rand(Dirichlet([1.0, 1.0]), 10)
            gs = GumbelSoftmax(p)
            xs = [rand(gs) for _ = 1:n]
            @test mean(xs) ≈ p atol=2ATOL_RAND
        end

        u = rand(5)
        f(x) = sum(u2gumbel(x))
        g_ad = gradient(f, u)
        g_fd = central_fdm(5, 1)(f, u)
        @test g_ad ≈ g_fd
    end

    # @testset "BatchGumbelBernoulli" begin
    #     for _ = 1:NUM_RANDTESTS
    #         p = Matrix{FT}(rand(Beta(1.0, 1.0), 1, 1))
    #
    #         gs2d = BatchGumbelSoftmax2D(AT(p))
    #         gb = BatchGumbelBernoulli(AT(p))
    #
    #         x = Array(hcat([rand(gs2d)[1,:] for _ = 1:n]...))
    #         # @info "" x p rand(gs2d)
    #         @test mean(x; dims=2) ≈ p atol=ATOL_RAND
    #
    #         x = Array([rand(gb)[1,1] for _ = 1:n])
    #         @test mean(x; dims=1) ≈ p atol=ATOL_RAND
    #         @test mean(x; dims=1) ≈ Array(mean(gb)) atol=ATOL_RAND
    #
    #         x = rand(gb)
    #         @test logpdf(gb, x) ≈ logpdfCoV(gb, x) atol=100ATOL
    #     end
    # end
    #
    # @testset "BatchGumbelBernoulliLogit" begin
    #     for _ = 1:NUM_RANDTESTS
    #         p = Matrix{FT}(rand(Beta(1.0, 1.0), 1, 1))
    #
    #         gbl = BatchGumbelBernoulliLogit(AT(logit.(p)))
    #
    #         logitx = Array([logitrand(gbl)[1] for _ = 1:n])
    #         x = logistic.(logitx)
    #         @test mean(x; dims=1) ≈ p atol=ATOL_RAND
    #         @test mean(x; dims=1) ≈ Array(mean(gbl)) atol=ATOL_RAND
    #     end
    #
    #     @warn "`logpdflogit(gbl::BatchGumbelBernoulliLogit, logitx)` is not tested."
    # end
end
