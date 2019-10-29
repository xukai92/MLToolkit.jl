using Test, MLToolkit
using MLToolkit.DistributionsX: u2gumbel
using Distributions: Dirichlet
using Statistics: mean
using Tracker: gradient, data
using FiniteDifferences: central_fdm
# using Distributions: Beta
# using StatsFuns: logit, logistic

@testset "Gumbel" begin
    n_randtests = 5
    n_samples = 5_000
    d, n = 2, 10
    atol = 0.015

    @testset "GumbelSoftmax" begin
        for _ = 1:n_randtests
            # Vector `p`
            p = rand(Dirichlet(ones(d)))
            gs = GumbelSoftmax(p)
            x = rand(gs, n_samples)
            @test mean(gs) == p
            @test vec(mean(x; dims=2)) ≈ mean(gs) atol=atol * d
            # Matrix `p`
            p = rand(Dirichlet(ones(d)), n)
            gs = GumbelSoftmax(p)
            xs = [rand(gs) for _ = 1:n_samples]
            @test mean(xs) ≈ mean(gs) atol=atol * d * n
        end

        @testset "GumbelSoftmax2D" begin
            # Vector `p`
            p = rand()
            gs = GumbelSoftmax2D(p)
            x = rand(gs, n_samples)
            @test mean(gs) == [p, 1 - p]
            @test vec(mean(x; dims=2)) ≈ mean(gs) atol=atol * 2
            # Matrix `p`
            p = rand(n)
            gs = GumbelSoftmax2D(p)
            xs = [rand(gs) for _ = 1:n_samples]
            @test mean(xs) ≈ mean(gs) atol=atol * 2 * n
        end

        @testset "track(u2gumbel, u)" begin
            u = rand()
            f(x) = sum(u2gumbel(x))
            g_ad, = data.(gradient(f, u))
            g_fd = central_fdm(5, 1)(f, u)
            @test g_ad ≈ g_fd
        end
    end

    # @testset "BatchGumbelBernoulli" begin
    #     for _ = 1:n_randtests
    #         p = Matrix{FT}(rand(Beta(1.0, 1.0), 1, 1))
    #
    #         gs2d = BatchGumbelSoftmax2D(AT(p))
    #         gb = BatchGumbelBernoulli(AT(p))
    #
    #         x = Array(hcat([rand(gs2d)[1,:] for _ = 1:n]...))
    #         # @info "" x p rand(gs2d)
    #         @test mean(x; dims=2) ≈ p atol=atol
    #
    #         x = Array([rand(gb)[1,1] for _ = 1:n])
    #         @test mean(x; dims=1) ≈ p atol=atol
    #         @test mean(x; dims=1) ≈ Array(mean(gb)) atol=atol
    #
    #         x = rand(gb)
    #         @test logpdf(gb, x) ≈ logpdfCoV(gb, x) atol=100ATOL
    #     end
    # end
    #
    # @testset "BatchGumbelBernoulliLogit" begin
    #     for _ = 1:n_randtests
    #         p = Matrix{FT}(rand(Beta(1.0, 1.0), 1, 1))
    #
    #         gbl = BatchGumbelBernoulliLogit(AT(logit.(p)))
    #
    #         logitx = Array([logitrand(gbl)[1] for _ = 1:n])
    #         x = logistic.(logitx)
    #         @test mean(x; dims=1) ≈ p atol=atol
    #         @test mean(x; dims=1) ≈ Array(mean(gbl)) atol=atol
    #     end
    #
    #     @warn "`logpdflogit(gbl::BatchGumbelBernoulliLogit, logitx)` is not tested."
    # end
end
