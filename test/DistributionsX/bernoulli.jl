using Test, MLToolkit
using Distributions: Beta
using Statistics: mean
using StatsFuns: logit, logistic
using Flux: gpu, use_cuda

@testset "Bernoulli" begin
    n_randtests = 5
    n_samples = 5_000
    d, n = 2, 10
    atol = 0.015

    @testset "Bernoulli" begin
        for _ = 1:n_randtests
            p = rand(d, n) |> gpu
            b = Bernoulli(p)
            @test mean(b) == p
            xs = [rand(b) for _ = 1:n_samples]
            @test mean(xs) ≈ mean(b) atol=atol * d * n

            @test logpdf.(Bernoulli.(p), xs[1]) ≈ logpdf(b, xs[1])

            b2 = Bernoulli(rand(d, n) |> gpu)
            kl_mc = mean(logpdf.(b, xs) - logpdf.(b2, xs))
            @test kldiv(b, b2) ≈ kl_mc atol=atol * d * n
        end

        @warn "`rand(b::Bernoulli)` is not tested."
    end

    # @testset "BatchBernoulliLogit" begin
    #     for _ = 1:n_randtests
    #         p = Matrix{FT}(rand(Beta(1.0, 1.0), 1, 1))
    #
    #         b = Bernoulli(p[1,1])
    #         bbl = BatchBernoulliLogit{AT}(logit.(p))
    #
    #         x = rand(b, n)
    #         @test cpucopy(vec(logpdf(bbl, AT{FT,2}(reshape(x, 1, n))))) ≈ logpdf.(b, x) atol=5ATOL
    #     end
    # end
end
