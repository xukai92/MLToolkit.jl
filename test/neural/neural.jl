using Test, MLToolkit.Neural

@testset "Neural" begin
    tests = [
        "architecture.jl",
    ]
    foreach(include, tests)

    @warn "`Neural.jl` is not tested."

    @warn "`gen.jl` is not tested."
end
