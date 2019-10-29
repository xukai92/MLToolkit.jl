using Test, MLToolkit
using MLToolkit: _lbeta, _beta

@testset "Special" begin
    n_tests = 5
    sz = (10, 5)

    @testset "lbeta" begin
        for _ = 1:n_tests
            x = randn(FT, sz...).^2
            y = randn(FT, sz...).^2
            @test _lbeta.(AT(x), AT(y)) ≈ lbeta.(x, y)
        end
    end

    @testset "beta" begin
        for _ = 1:n_tests
            x = randn(FT, sz...).^2
            y = randn(FT, sz...).^2
            @test _beta.(AT(x), AT(y)) ≈ beta.(x, y)
        end
    end

    @warn "`logit` is not tested."
    @warn "`logsumexp` is not tested."
end
