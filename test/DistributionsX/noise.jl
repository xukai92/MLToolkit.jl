using Test, MLToolkit
using Distributions: Normal
using MLToolkit.DistributionsX: test_stat
using Statistics: mean, var
using Flux: gpu

@testset "Noise" begin
    dim1, dim2 = 10, 5
    n_samples, atol = 5_000, 0.01

    @testset "UniformNoise" begin
        noise = UniformNoise(dim1) |> gpu
        @test size(noise) == (dim1,)
        @test size(rand(noise)) == size(noise)
        @test size(rand(noise, dim2)) == (size(noise)..., dim2)
        @test size(rand(UniformNoise(dim1, dim2))) == (dim1, dim2)

        x = test_stat(mean, noise, n_samples, atol * prod(size(noise)))
        test_stat(var, noise, x, atol * prod(size(noise)))

        @test logpdf(noise, x) ≈ log.(ones(dim1, n_samples) / 2)
    end

    @testset "GaussianNoise" begin
        noise = GaussianNoise(dim1) |> gpu
        @test size(noise) == (dim1,)
        @test size(rand(noise)) == size(noise)
        @test size(rand(noise, dim2)) == (size(noise)..., dim2)
        @test size(rand(GaussianNoise(dim1, dim2))) == (dim1, dim2)

        x = test_stat(mean, noise, n_samples, atol * prod(size(noise)))
        test_stat(var, noise, x, atol * prod(size(noise)))

        @test logpdf(noise, x) ≈ map(xi -> logpdf(Normal(0, 1), xi), x)
    end
end
