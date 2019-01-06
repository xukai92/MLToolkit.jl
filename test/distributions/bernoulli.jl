using MLToolkit, Test
using Distributions: Beta, Bernoulli
using Statistics: mean

@testset "Bernoulli" begin
    n = 100_000
    
    @testset "BatchBernoulli" begin
        for _ = 1:NUM_RANDTESTS
            p = Matrix{FT}(rand(Beta(1.0, 1.0), 1, 1))

            b = Bernoulli(p[1,1])
            bb = BatchBernoulli{AT}(p)

            x = rand(b, n)
            @test vec(logpdf(bb, reshape(x, 1, n))) ≈ logpdf.(b, x) atol=ATOL

            q = Matrix{FT}(rand(Beta(1.0, 1.0), 1, 1))
            b2 = Bernoulli(q[1,1])
            bb2 = BatchBernoulli{AT}(q)

            kl_12 = mean(logpdf.(b, x) - logpdf.(b2, x))

            @test sum(kl(bb, bb2)) ≈ kl_12 atol=ATOL_RAND
        end
    end
end
