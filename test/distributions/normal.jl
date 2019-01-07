using Test, MLToolkit
using Distributions: MvNormal, Beta
using Statistics: mean, var
using LinearAlgebra: I

@testset "Normal" begin
    d = 10
    n = 500_000

    @testset "BatchNormal" begin
        for _ = 1:NUM_RANDTESTS
            # rand
            μ = randn(FT, d, 1); Σ = randn(FT, d, 1).^2

            bn = BatchNormal{AT}(μ, Σ)
            x = Array(hcat([rand(bn) for _ = 1:n]...))

            @test mean(x; dims=2) ≈ μ atol=(d * ATOL_RAND)
            @test var(x; dims=2) ≈ Σ atol=(d * ATOL_RAND)

            # logpdf
            mvn = MvNormal(vec(μ), sqrt.(vec(Σ)))
            x = Matrix{FT}(rand(mvn, n))
            lp = logpdf(mvn, x)

            @test vec(sum(logpdf(bn, AT(x)); dims=1)) ≈ lp atol=(d * ATOL)

            # kl
            μ1 = zeros(FT, d, 1); Σ1 = ones(FT, d, 1)
            μ2 = μ1 .+ rand(FT); Σ2 = Σ1 .* abs(rand(FT))

            mvn1 = MvNormal(vec(μ1), sqrt.(vec(Σ1)))
            mvn2 = MvNormal(vec(μ2), sqrt.(vec(Σ2)))
            x = rand(mvn1, n)
            kl_mc = mean(logpdf(mvn1, x) - logpdf(mvn2, x))

            bn1 = BatchNormal{AT}(μ1, Σ1)
            bn2 = BatchNormal{AT}(μ2, Σ2)
            @test sum(kldiv(bn1, bn2)) ≈ kl_mc atol=(d * ATOL_RAND)

            @test sum(kldiv(bn1, BatchNormal(μ2[1,1], Σ2[1,1]))) ≈ kl_mc atol=(d * ATOL_RAND)
        end
    end

    # NOTE: the test below doesn't include `KnetArray` because the lack of
    #       the support of `det` and `inv` for `KnetArray`.
    @testset "MvNormal" begin
        for _ = 1:NUM_RANDTESTS
            μ1 = zeros(FT, d); Σ1 = Matrix{FT}(I, d, d)
            μ2 = μ1 .+ rand(FT); Σ2 = Σ1 .* abs(rand(FT))

            mvn1 = MvNormal(μ1, Σ1); mvn2 = MvNormal(μ2, Σ2)
            x = rand(mvn1, n)

            kl_12 = mean(logpdf(mvn1, x) - logpdf(mvn2, x))
            @test kldiv(mvn1, mvn2) ≈ kl_12 atol=(d * ATOL_RAND)
        end
    end
end
