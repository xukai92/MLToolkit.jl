using MLToolkit, Test
using Distributions: Beta
using Statistics: mean

@testset "Beta" begin
    n = 5_000
    @testset "Kumaraswamy" begin
        for (a, b) = [(1.0, 1.0), (1.0, 2.0), (2.0, 1.0)]
            beta_mean = mean(rand(Beta(a, b), n))

            a_mat = Matrix{FT}(undef, 1, 1)
            b_mat = Matrix{FT}(undef, 1, 1)
            a_mat[1,1] = a
            b_mat[1,1] = b
            kuma = Kumaraswamy{AT}(a_mat, b_mat)

            x = Array(hcat([rand(kuma) for _ = 1:n]...))
            @test mean(x) ≈ beta_mean atol=ATOL_RAND

            x = Array(hcat([exp.(logrand(kuma)) for _ = 1:n]...))
            @test mean(x) ≈ beta_mean atol=ATOL_RAND

            # x = Array(hcat([sample_from_kumaraswamy_iid(AT(a_mat), AT(b_mat), 1, 1) for _ = 1:n]...))
            # p_kumaraswamy_mean = mean(x; dims=2)
            # @test beta_mean ≈ p_kumaraswamy_mean[1] atol=ATOL_RAND
        end
    end

    # @testset "Beta distributions" begin
    #     # Test cases are taken from https://en.wikipedia.org/wiki/Beta_distribution
    #     for f = [kl_beta, kl_beta_slow]
    #         @test f(1.0, 1.0, 3.0, 3.0) ≈ 0.598803 atol=ATOL_DEFAULT
    #         @test f(3.0, 3.0, 1.0, 1.0) ≈ 0.267864 atol=ATOL_DEFAULT
    #         @test f(3.0, 0.5, 0.5, 3.0) ≈ 7.21574 atol=ATOL_DEFAULT
    #         @test f(0.5, 3.0, 3.0, 0.5) ≈ 7.21574 atol=ATOL_DEFAULT
    #     end
    # end
    #
    # @testset "Kumaraswamy Beta" begin
    #     for rand_test_id = 1:RAND_TEST_NUM
    #         (a, b, α, β) = 1.0 .+ rand(4).^2
    #
    #         a_mat, b_mat = Matrix{FT}(undef, 1, 1), Matrix{FT}(undef, 1, 1)
    #         a_mat[1] = a; b_mat[1] = b
    #
    #         x = hcat([Array(sample_from_kumaraswamy(AT(a_mat), AT(b_mat))) for _ = 1:5000]...)
    #         kl_k_b = mean(logpdf_kumaraswamy(a, b, x) - logpdf.(Beta(α, β), x))
    #         @test kl_kumaraswamy_beta(a, b, α, β) ≈ kl_k_b atol=ATOL_RAND
    #     end
    # end
end
