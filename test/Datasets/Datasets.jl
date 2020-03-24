using Test, MLToolkit.Datasets

@testset "Datasets" begin
    IGNORES = ["mnist", "cifar10"]

    for name in Datasets.DATASET_NAMES
        name in IGNORES && continue
        dataset = Dataset(name, 30_000)
    end

    @warn "Datasets in $IGNORES are not tested."
end
