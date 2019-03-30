using Test
using MLToolkit: include_list_as_module

@testset "Neural" begin
    @warn "`neural.jl` is not tested."

    tests = [
        "rho.jl",
        "sbc.jl",
        "layers.jl",
        "stochastic_layers.jl",
        "activations.jl",
    ]

    include_list_as_module(tests, "TestNeural")
end
