using Test, MLToolkit.Neural

@testset "Neural" begin
    tests = [
        # "architecture.jl",
    ]
    foreach(include, tests)

    @warn "`Datasets.jl` is not tested."
end
