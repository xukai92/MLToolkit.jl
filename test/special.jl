using MLToolkit, Test
using MLToolkit: _lbeta, _beta

@testset "Special" begin
    dims = (10, 5)

    @testset "lbeta" begin
        for _ = 1:NUM_RANDTESTS
            x = AT(randn(FT, dims...).^2)
            y = AT(randn(FT, dims...).^2)
            @test _lbeta.(x, y) ≈ lbeta.(x, y)
        end
    end

    @testset "beta" begin
        for _ = 1:NUM_RANDTESTS
            x = AT(randn(FT, dims...).^2)
            y = AT(randn(FT, dims...).^2)
            @test _beta.(x, y) ≈ beta.(x, y)
        end
    end
end
