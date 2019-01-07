using Test, MLToolkit

@testset "Activations" begin
    dims = (10, 5)

    @testset "softplus" begin
        for _ = 1:10
            x = randn(dims...)
            y = softplus.(x)
            @test all(y .> 0)

            exp_y = exp_softplus.(x)
            @test all(y .== log.(exp_y))
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
