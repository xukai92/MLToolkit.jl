using MLToolkit, Test
using Distributions: Poisson, MvNormal, Beta, Dirichlet, Bernoulli
using Statistics: mean, var
using Knet: gpu, KnetArray
using LinearAlgebra: I

const NUM_RANDOM_TESTS = 5
const ATOL_DEFAULT = 1e-6
const ATOL_RAND = 1e-2
const FT = Float64
# Test on GPU whenever possible
const AT = gpu() != -1 ? KnetArray : Array

@testset "Distributions" begin
    @testset "DisplacedPoisson" begin
        for _ = 1:NUM_RANDOM_TESTS
            λ = rand() * 10

            p = Poisson(λ)

            x = 0:1:100
            y_p = pdf.(Ref(p), x)
            # When r equals 0, displaced Poisson reduces to normal Poisson.
            # Thus we can check the p.d.f using the Poisson (from Distributions.jl).
            dpeqv = DisplacedPoisson(λ, 0.0)
            y_dpeqv = pdf.(Ref(dpeqv), x)
            @test all(isapprox.(y_p, y_dpeqv))

            m_p = mode(p)
            m_dpeqv = mode(dpeqv)
            @test m_p == m_dpeqv
        end
    end

    @testset "IBP" begin
        d, n = 10, 5

        ibp = IBP(2.0)
        Z = rand(ibp, d, n)
        @test size(Z) == (n, d)
    end

    @testset "Normal" begin
        d, n = 10, 500_000

        @testset "BatchNormal" begin
            for _ = 1:NUM_RANDOM_TESTS
                # rand
                μ = randn(FT, d, 1); Σ = randn(FT, d, 1).^2

                bn = BatchNormal{AT}(μ, Σ)
                x = Array(hcat([rand(bn) for _ = 1:n]...))

                @test mean(x; dims=2) ≈ μ atol=(d * ATOL_RAND)
                @test var(x; dims=2) ≈ Σ atol=(d * ATOL_RAND)

                # logpdf
                mvn = MvNormal(vec(μ), sqrt.(vec(Σ)))
                x = rand(mvn, n)
                lp = logpdf(mvn, x)

                @test vec(logpdf(bn, x)) ≈ lp atol=(d * ATOL_DEFAULT)

                # kl
                μ1 = zeros(FT, d, 1); Σ1 = ones(FT, d, 1)
                μ2 = μ1 .+ rand(FT); Σ2 = Σ1 .* abs(rand(FT))

                mvn1 = MvNormal(vec(μ1), sqrt.(vec(Σ1)))
                mvn2 = MvNormal(vec(μ2), sqrt.(vec(Σ2)))
                x = rand(mvn1, n)

                kl_12 = mean(logpdf(mvn1, x) - logpdf(mvn2, x))

                bn1 = BatchNormal{AT}(μ1, Σ1)
                bn2 = BatchNormal{AT}(μ2, Σ2)
                @test sum(kl(bn1, bn2)) ≈ kl_12 atol=(d * ATOL_RAND)

                @test sum(kl(bn1, BatchNormal(μ2[1,1], Σ2[1,1]))) ≈ kl_12 atol=(d * ATOL_RAND)
            end
        end

        # NOTE: the test below is not for `KnetArray` because the lack of
        #       the support of `det` and `inv` for `KnetArray`.
        @testset "MvNormal" begin
            for _ = 1:NUM_RANDOM_TESTS
                μ1 = zeros(FT, d); Σ1 = Matrix{FT}(I, d, d)
                μ2 = μ1 .+ rand(FT); Σ2 = Σ1 .* abs(rand(FT))

                mvn1 = MvNormal(μ1, Σ1); mvn2 = MvNormal(μ2, Σ2)
                x = rand(mvn1, n)

                kl_12 = mean(logpdf(mvn1, x) - logpdf(mvn2, x))
                @test kl(mvn1, mvn2) ≈ kl_12 atol=(d * ATOL_RAND)
            end
        end
    end

    @testset "Bernoulli" begin
        n = 100_000
        @testset "BatchBernoulli" begin
            for _ = 1:NUM_RANDOM_TESTS
                p = Matrix{FT}(rand(Beta(1.0, 1.0), 1, 1))

                b = Bernoulli(p[1,1])
                bb = BatchBernoulli{AT}(p)

                x = rand(b, n)
                @test vec(logpdf(bb, reshape(x, 1, n))) ≈ logpdf.(b, x) atol=ATOL_DEFAULT

                q = Matrix{FT}(rand(Beta(1.0, 1.0), 1, 1))
                b2 = Bernoulli(q[1,1])
                bb2 = BatchBernoulli{AT}(q)

                kl_12 = mean(logpdf.(b, x) - logpdf.(b2, x))

                @test sum(kl(bb, bb2)) ≈ kl_12 atol=ATOL_RAND
            end
        end
    end

    @testset "Gumbel" begin
        τ_atol_ratio = 2
        n = 100_000

        @testset "GumbelSoftmax" begin
            for _ = 1:NUM_RANDOM_TESTS
                p = Matrix{FT}(rand(Dirichlet([1.0, 1.0]), 1))

                gs = GumbelSoftmax{AT}(p)
                x = hcat([rand(gs) for _ = 1:n]...)

                @test mean(x; dims=2) ≈ p atol=(2 * τ_atol_ratio * ATOL_RAND)
            end
        end

        @testset "GumbelBernoulli" begin
            for _ = 1:NUM_RANDOM_TESTS
                p = Matrix{FT}(rand(Beta(1.0, 1.0), 1, 1))
                gs2d = GumbelSoftmax2D{AT}(p)
                gb = GumbelBernoulli{AT}(p)

                x = Array(hcat([rand(gs2d)[1,:] for _ = 1:n]...))
                @test mean(x; dims=2) ≈ p atol=ATOL_RAND

                x = Array([rand(gb)[1,1] for _ = 1:n])
                @test mean(x; dims=1) ≈ p atol=ATOL_RAND

            end
        end

        # @testset "GumbelBernoulliLogit" begin
            # p = Matrix{FT}(rand(Beta(1.0, 1.0), 1, 1))
            # lp = log.(p ./ (1 .- p))
            # lx = Array([sample_logit_from_bernoulli(AT(lp))[1] for _ = 1:5000])
            # x = 1 ./ (1 .+ exp.(-lx))
            # p_est = mean(x; dims=1)
            # @test p_est ≈ p atol=ATOL_RAND
        # end
    end
end
