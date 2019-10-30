using Test, MLToolkit
using Statistics: mean, var

@testset "Noise" begin
    d, n = 10, 5_000
    atol = 0.01 * d

    @testset "UniformNoise" begin
        noise = UniformNoise(d)
        x = rand(noise, n)
        @test vec(mean(x; dims=2)) ≈ mean(noise) atol=atol
        @test vec(var(x; dims=2)) ≈ var(noise) atol=atol

        @test logpdf(noise, x) ≈ log.((ones(n) / 2) .^ d)
    end

    @testset "GaussianNoise" begin
        noise = GaussianNoise(d)
        x = rand(noise, n)
        @test vec(mean(x; dims=2)) ≈ mean(noise) atol=atol
        @test vec(var(x; dims=2)) ≈ var(noise) atol=atol

        # @test logpdf(noise, x)    == ones(n) / 2
    end
end
