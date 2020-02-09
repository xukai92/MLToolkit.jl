using Test, MLToolkit.Datasets

@testset "Neural" begin
    tests = [
        # "architecture.jl",
    ]
    foreach(include, tests)

    for name in Datasets.DATASET_NAMES
        dataset = Dataset(name, 30_000)
    end
end
