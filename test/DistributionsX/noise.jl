using Test, MLToolkit
using DistributionsAD: TuringDiagNormal
using MLToolkit.DistributionsX: test_stat
using Statistics: mean, var

@testset "Noise" begin
    d, n_samples = 10, 5_000
    atol = 0.01 * d

    @testset "UniformNoise" begin
        noise = UniformNoise(d)
        x = test_stat(mean, noise, n_samples, atol)
        test_stat(var, noise, x, atol)

        @test logpdf(noise, x) ≈ log.((ones(n_samples) / 2) .^ d)
    end

    @testset "GaussianNoise" begin
        noise = GaussianNoise(d)
        x = test_stat(mean, noise, n_samples, atol)
        test_stat(var, noise, x, atol)

        @test logpdf(noise, x) ≈ logpdf(TuringDiagNormal(zeros(d), ones(d)), x)
    end
end
