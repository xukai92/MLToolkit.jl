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

        for train in [
            (x, y),
            (x=x, y=y)
        ]
            dataset = Dataset(train)

            batch_size = 20
            loader = DataLoader(dataset, batch_size; withidx=true)
            @test length(loader.train) == 5
            x1, y1 = first(loader.train).data
            @test size(x1, 2) == batch_size
            @test size(y1, 1) == batch_size
        end
    end
end
