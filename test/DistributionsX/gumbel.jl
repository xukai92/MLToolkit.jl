using Test, MLToolkit
using MLToolkit.DistributionsX: u2gumbel
using Distributions: Dirichlet
using Statistics: mean
using Tracker: gradient, data
using FiniteDifferences: central_fdm
using StatsFuns: logit, logistic
using Flux: gpu

@testset "Gumbel" begin
    n_randtests = 5
    n_samples = 5_000
    d, n = 2, 10
    atol = 0.015

    @testset "GumbelSoftmax" begin
        for _ = 1:n_randtests
            # Vector `p`
            p = rand(Dirichlet(ones(d))) |> gpu
            gs = GumbelSoftmax(p)
            @test mean(gs) == p
            x = rand(gs, n_samples)
            @test vec(mean(x; dims=2)) ≈ mean(gs) atol=atol * d
            # Matrix `p`
            p = rand(Dirichlet(ones(d)), n) |> gpu
            gs = GumbelSoftmax(p)
            xs = [rand(gs) for _ = 1:n_samples]
            @test mean(xs) ≈ mean(gs) atol=atol * d * n
        end
    end

    @testset "GumbelSoftmax2D" begin
        for _ = 1:n_randtests
            # Vector `p`
            p1 = rand()
            gs = GumbelSoftmax2D(p1)
            @test mean(gs) == [p1, 1 - p1]
            x = rand(gs, n_samples)
            @test vec(mean(x; dims=2)) ≈ mean(gs) atol=atol * 2
            # Matrix `p`
            p1 = rand(n) |> gpu
            gs = GumbelSoftmax2D(p1)
            @test mean(gs) == [p1'; 1 .- p1']
            xs = [rand(gs) for _ = 1:n_samples]
            @test mean(xs) ≈ mean(gs) atol=atol * 2 * n
        end
    end

    @testset "track(u2gumbel, u)" begin
        for _ = 1:n_randtests
            u = rand() / 2 + 1 / 4
            f(x) = sum(u2gumbel(x))
            g_ad, = data.(gradient(f, u))
            g_fd = central_fdm(5, 1)(f, u)
            @test g_ad ≈ g_fd
        end
    end

    @testset "GumbelBernoulli" begin
        for _ = 1:n_randtests
            p = rand(d, n) |> gpu
            gb = GumbelBernoulli(p)
            @test mean(gb) == p
            xs = [rand(gb) for _ = 1:n_samples]
            @test mean(xs) ≈ mean(gb) atol=atol * d * n
            # Consistency between two versions of log density
            x = rand(gb)
            @test logpdf(gb, x) ≈ logpdfCoV(gb, x)
        end
    end

    @testset "GumbelBernoulliLogit" begin
        for _ = 1:n_randtests
            p = rand(d, n) |> gpu
            gbl = GumbelBernoulliLogit(logit.(p))
            @test mean(gbl) ≈ p
            xs = [rand(gbl) for _ = 1:n_samples]
            @test mean(xs) ≈ mean(gbl) atol=atol * d * n
        end
    end
end
