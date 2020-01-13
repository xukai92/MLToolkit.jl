using Test, MLToolkit.DistributionsX
using MLToolkit: seed!
using Distributions: Normal, MvNormal
using Statistics: mean, var, std
using Flux: gpu
seed!(1)

@testset "Normal" begin
    n_randtests = 3
    n_samples = 10_000
    d, n = 5, 10
    rtol = 0.02

    @testset "BroadcastedNormal" begin
        for _ = 1:n_randtests
            # Construction
            m, s = randn(d, n) |> gpu, rand(d, n) |> gpu
            bd1 = BroadcastedNormal(m, s)
            # Statistics
            @test mean(bd1) == m
            @test std(bd1)  == s
            # Sample
            xs = [rand(bd1) for _ = 1:n_samples]
            @test mean(xs) ≈ mean(bd1) rtol=rtol
            @test std(xs)  ≈ std(bd1)  rtol=rtol
            # Sample (multi)
            X = rand(bd1, n_samples)
            @test mean(X; dims=3) ≈ m rtol=rtol
            @test  std(X; dims=3) ≈ s rtol=rtol
            # Density
            @test logpdf.(Normal.(m, s), X) ≈ logpdf(bd1, X)
            # Density (multi)
            X1 = X[:,:,1]
            m1, s1 = m[:,1], s[:,1]
            @test logpdf(MvNormal(m1, s1), X1) ≈ sum(logpdf(BroadcastedNormal(m1, s1), X1), true; dims=1)
            # KL divergence
            m, s = (m, s) .+ (randn(d, n) |> gpu, rand(d, n) |> gpu)
            bd2 = BroadcastedNormal(m, s)
            kl_mc = mean(logpdf.(bd1, xs) - logpdf.(bd2, xs))
            @test kldiv(bd1, bd2) ≈ kl_mc rtol=rtol
        end
    end

    @testset "BroadcastedNormalLogStd" begin
        for _ = 1:n_randtests
            # Construction
            m, s = randn(d, n) |> gpu, rand(d, n) |> gpu
            bd1 = BroadcastedNormalLogStd(m, log.(s))
            # Statistics
            @test mean(bd1) == m
            @test std(bd1)  ≈  s
            # Sample
            xs = [rand(bd1) for _ = 1:n_samples]
            @test mean(xs) ≈ mean(bd1) rtol=rtol
            @test std(xs)  ≈ std(bd1)  rtol=rtol
            # Sample (multi)
            X = rand(bd1, n_samples)
            @test mean(X; dims=3) ≈ m rtol=rtol
            @test  std(X; dims=3) ≈ s rtol=rtol
            # Density
            @test logpdf.(Normal.(m, s), X) ≈ logpdf(bd1, X)
            # Density (multi)
            X1 = X[:,:,1]
            m1, s1 = m[:,1], s[:,1]
            @test logpdf(MvNormal(m1, s1), X1) ≈ dropdims(sum(logpdf(BroadcastedNormalLogStd(m1, log.(s1)), X1); dims=1); dims=1)
            # KL divergence
            m, s = (m, s) .+ (randn(d, n) |> gpu, rand(d, n) |> gpu)
            bd2 = BroadcastedNormalLogStd(m, log.(s))
            kl_mc = mean(logpdf.(bd1, xs) - logpdf.(bd2, xs))
            @test kldiv(bd1, bd2) ≈ kl_mc rtol=rtol
        end
    end

    # using LinearAlgebra: I

    # @testset "MvNormal" begin
    #     for _ = 1:NUM_RANDTESTS
    #         μ1 = zeros(FT, d); Σ1 = Matrix{FT}(I, d, d)
    #         μ2 = μ1 .+ rand(FT); Σ2 = Σ1 .* abs(rand(FT))

    #         mvn1 = MvNormal(μ1, Σ1); mvn2 = MvNormal(μ2, Σ2)
    #         x = rand(mvn1, n)

    #         kl_12 = mean(logpdf(mvn1, x) - logpdf(mvn2, x))
    #         @test kldiv(mvn1, mvn2) ≈ kl_12 atol=(d * ATOL_RAND)
    #     end
    # end
end
