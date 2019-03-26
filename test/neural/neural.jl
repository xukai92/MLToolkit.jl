using Test
using MLToolkit: include_list_as_module

@testset "Neural" begin
    @warn "`neural.jl` is not tested."

    tests = [
        "rho",
        "layers",
        "stochastic_layers",
        "activations",
    ]

    include_list_as_module(tests, "TestNeural")
end
