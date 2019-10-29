using Test, MLToolkit
using Statistics: mean, var

@testset "Noise" begin
    d, n = 10, 5_000
    atol = 0.01 * d

    @testset "UniformNoise" begin
        noise = UniformNoise(d)
        x = rand(noise, n)
        @test vec(mean(x; dims=2)) ≈ zeros(d) atol=atol
        @test vec(var(x; dims=2)) ≈ ones(d) / 3 atol=atol
    end

    @testset "GaussianNoise" begin
        noise = GaussianNoise(d)
        x = rand(noise, n)
        @test vec(mean(x; dims=2)) ≈ zeros(d) atol=atol
        @test vec(var(x; dims=2)) ≈ ones(d) atol=atol
    end
end
