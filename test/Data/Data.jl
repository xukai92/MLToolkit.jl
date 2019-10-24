using Test, MLToolkit.Data

@testset "Data" begin
    @testset "get_features" begin
        @warn "`get_features_griffiths2011indian()` not tested"
        @warn "`get_features_large()` not tested"
    end

    @testset "Ring" begin
        @warn "`Ring` not tested"
        @warn "`makemixturemodel()` not tested"
    end

    @testset "Dataset and DataLoader" begin
        n = 100
        x = randn(784, n)
        y = rand(n)

        dataset = Dataset((x, y))

        batch_size = 20
        loader = DataLoader(dataset, batch_size)
        @test length(loader.train) == 5
        x1, y1 = first(loader.train).batch
        @test size(x1, 2) == batch_size
        @test size(y1, 1) == batch_size
    end
end
