using Test, MLToolkit
using Distributions: Beta, Bernoulli
using Statistics: mean
using StatsFuns: logit

@testset "Bernoulli" begin
    n = 100_000

    @testset "BatchBernoulli" begin
        for _ = 1:NUM_RANDTESTS
            p = Matrix{FT}(rand(Beta(1.0, 1.0), 1, 1))

            b = Bernoulli(p[1,1])
            bb = BatchBernoulli{AT}(p)

            x = rand(b, n)
            @test vec(logpdf(bb, AT{FT,2}(reshape(x, 1, n)))) ≈ logpdf.(b, x) atol=ATOL

            q = Matrix{FT}(rand(Beta(1.0, 1.0), 1, 1))
            b2 = Bernoulli(q[1,1])
            bb2 = BatchBernoulli{AT}(q)

            kl_mc = mean(logpdf.(b, x) - logpdf.(b2, x))

            @test sum(kldiv(bb, bb2)) ≈ kl_mc atol=ATOL_RAND
        end

        @warn "`rand(bb::BatchBernoulli)` is not tested."
    end

    @testset "BatchBernoulliLogit" begin
        for _ = 1:NUM_RANDTESTS
            p = Matrix{FT}(rand(Beta(1.0, 1.0), 1, 1))

            b = Bernoulli(p[1,1])
            bbl = BatchBernoulliLogit{AT}(logit.(p))

            x = rand(b, n)
            @test vec(logpdf(bbl, AT{FT,2}(reshape(x, 1, n)))) ≈ logpdf.(b, x) atol=ATOL
        end
    end
end
