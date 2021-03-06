using Test, Flux, MLToolkit.Neural

@testset "Neural" begin
    @warn "`Neural.jl` is not tested."

    @testset "Architecture" begin
        σ, σlast = x -> sin.(x), x -> x
        Dout, B = 10, 5

        @testset "DenseNet" begin
            Din = 50
            mlp = DenseNet(Din, Dout, σ, σlast)
            Dhs = (40, 30, 20)
            mlp = DenseNet(Din, Dhs, Dout, σ, σlast)
            @test_throws DimensionMismatch mlp(randn(Din - 1, B))
            x = randn(Din, B)
            @test size(mlp(x)) == (Dout, B)
            x = randn(5, 5, 2, B)
            @test size(mlp(x)) == (Dout, B)
        end

        @testset "ConvNet" begin
            W, H, C = 28, 28, 1
            @test_throws ErrorException ConvNet((W - 1, H - 1, C), Dout, σ, σlast)
            convnet = ConvNet((W, H, C), Dout, σ, σlast)

            @test_throws DimensionMismatch convnet(randn(W - 1, H, C, B))
            @test_throws DimensionMismatch convnet(randn(W, H - 1, C, B))
            @test_throws DimensionMismatch convnet(randn(W, H, C + 1, B))
            for x in [
                randn(W, H, C, B),
                randn(W * H * C, B)
            ]
                @test size(convnet(x)) == (Dout, B)
            end
        end
    end
end
