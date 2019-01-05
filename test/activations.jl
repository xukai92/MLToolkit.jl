using MLToolkit, Test

dims = (10, 10)

@testset "Activations" begin
    @testset "softplus" begin
        for _ = 1:10
            x = randn(dims...)
            y = softplus.(x)
            @test all(y .> 0)
        end
    end
    @testset "leaky_relu" begin
        for _ = 1:10
            x = randn(dims...)
            y = leaky_relu.(x; alpha=0)
            @test all(y .>= 0)
        end
    end
end
