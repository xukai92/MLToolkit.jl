using Test, MLToolkit
using Distributions: Beta, Dirichlet
using Statistics: mean

@testset "Beta" begin
    d = 10
    n = 500_000

    @testset "Kumaraswamy" begin
        for (a, b) = [(1.0, 1.0), (1.0, 2.0), (2.0, 1.0)]
            beta_mean = mean(rand(Beta(a, b), n))

            a_mat = Matrix{FT}(undef, 1, 1)
            b_mat = Matrix{FT}(undef, 1, 1)
            a_mat[1,1] = a
            b_mat[1,1] = b
            kuma = BatchKumaraswamy{AT}(a_mat, b_mat)

            x = Array(hcat([rand(kuma) for _ = 1:n]...))
            @test mean(x) ≈ beta_mean atol=ATOL_RAND

            x = Array(hcat([exp.(logrand(kuma)) for _ = 1:n]...))
            @test mean(x) ≈ beta_mean atol=ATOL_RAND

            kuma2 = BatchKumaraswamy(a, b)

            x = Array(rand(kuma2, n))
            @test mean(x) ≈ beta_mean atol=ATOL_RAND

            x = Array(exp.(logrand(kuma2, n)))
            @test mean(x) ≈ beta_mean atol=ATOL_RAND
        end
        @warn "`logpdf(kuma::BatchKumaraswamy, x)` is not tested."
    end

    @testset "Dirichlet" begin
        for _  = 1:NUM_RANDTESTS
            dir1 = Dirichlet(rand(FT, d).^2 .+ ones(FT, d))
            dir2 = Dirichlet(rand(FT, d).^2 .+ ones(FT, d))

            x = rand(dir1, n)

            kl_12 = mean(logpdf(dir1, x) - logpdf(dir2, x))

            @test kldiv(dir1, dir2) ≈ kl_12 atol=(d * ATOL_RAND)
        end
    end

    @testset "Beta" begin
        # Test cases are taken from https://en.wikipedia.org/wiki/Beta_distribution
        for (a1, b1, a2, b2, ans) = [(1.0, 1.0, 3.0, 3.0, 0.598803),
                                     (3.0, 3.0, 1.0, 1.0, 0.267864),
                                     (3.0, 0.5, 0.5, 3.0, 7.21574),
                                     (0.5, 3.0, 3.0, 0.5, 7.21574)]
            @test kldiv(BatchBeta(a1, b1), BatchBeta(a2, b2)) ≈ ans atol=100ATOL
            @test kldiv(Dirichlet([a1, b1]), Dirichlet([a2, b2])) ≈ ans atol=100ATOL
        end


        param_list = []
        for _ = 1:NUM_RANDTESTS
            (a1, b1) = 1.0 .+ rand(2).^2
            (a2, b2) = [a1, b1] .+ rand(2).^2
            push!(param_list, (a1, b1, a2, b2))
        end
        @testset "($a1,$b1) v.s. ($a2,$b2)" for (a1, b1, a2, b2) in param_list
            # Generate a1, b1, a2, b2 that are not too far away from each other
            (a1, b1) = 1.0 .+ rand(2).^2
            (a2, b2) = [a1, b1] .+ rand(2).^2

            @test kldiv(Dirichlet([a1, b1]), Dirichlet([a2, b2])) ≈
                  kldiv(BatchBeta(a1, b1), BatchBeta(a2, b2)) atol=ATOL_RAND

            kuma = BatchKumaraswamy(a1, b1)
            bb = BatchBeta(a2, b2)

            x = vec([rand(kuma) for _ = 1:n])
            kl_mc = mean(logpdf(kuma, x) - logpdf.(Beta(a2, b2), x))
            @test kldiv(kuma, bb) ≈ kl_mc atol=(2 * ATOL_RAND)
        end

        @warn "`logpdf(bb::BatchBeta, x)` is not tested."
    end
end
